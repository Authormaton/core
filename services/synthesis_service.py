"""
Service for synthesizing answers from ranked evidence passages.
Generates Markdown answers with citations and references.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any

from openai import OpenAI
from services.ranking_service import RankedEvidence
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

@dataclass
class SynthesisResult:
    """Result of synthesis with answer and metadata."""
    answer_markdown: str
    used_citation_ids: Set[int]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    generation_ms: int = 0

class SynthesisService:
    """
    Service for synthesizing answers from ranked evidence passages.
    Generates answers with citation markers and reference lists.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o"):
        """
        Initialize the synthesis service.
        
        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: LLM model to use
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required but not provided")
        
        self.model = model
        # Set a sane default timeout; endpoint also wraps with asyncio.wait_for
        self.client = AsyncOpenAI(api_key=self.api_key, timeout=30.0)
    def _build_system_prompt(self) -> str:
        """Build the system prompt for the synthesis."""
        return (
            "You are a factual technical writer. Your job is to answer questions based ONLY on the provided evidence. "
            "Be accurate, clear, and concise. Never include information not supported by the evidence."
        )
    
    def _build_evidence_block(self, evidence_list: List[RankedEvidence]) -> str:
        """
        Build a numbered evidence block with metadata.
        
        Args:
            evidence_list: List of ranked evidence
            
        Returns:
            Formatted evidence block
        """
        if not evidence_list:
            return "NO EVIDENCE PROVIDED."
        
        evidence_block = "# EVIDENCE\n\n"
        
        for evidence in evidence_list:
            # Format title
            title = evidence.title or "Untitled Document"
            
            # Format site info
            site_info = f" ({evidence.site_name})" if evidence.site_name else ""
            
            # Format evidence block
            evidence_block += f"[{evidence.id}] {title}{site_info}\n"
            evidence_block += f"URL: {evidence.url}\n"
            if evidence.published_at:
                evidence_block += f"Published: {evidence.published_at}\n"
            evidence_block += f"Excerpt: {evidence.passage}\n\n"
        
        return evidence_block
    
    def _build_user_prompt(self, query: str, evidence_list: List[RankedEvidence], 
                         answer_tokens: int, style_profile_id: Optional[str]) -> str:
        """
        Build the user prompt with query, evidence, and instructions.
        
        Args:
            query: The query to answer
            evidence_list: List of ranked evidence
            answer_tokens: Target token count for the answer
            style_profile_id: Optional style profile ID for tone/voice
            
        Returns:
            Complete user prompt
        """
        # Build evidence block
        evidence_block = self._build_evidence_block(evidence_list)
        
        # Build instructions
        instructions = (
            f"Question: {query}\n\n"
            "Instructions:\n"
            "1. Answer the question using ONLY the provided evidence.\n"
            "2. Add citation markers [^i] after each sentence, where i is the evidence number.\n"
            "3. If a claim lacks evidence, either mark it (citation needed) or omit it entirely.\n"
            "4. End with a References section listing each source you actually cited:\n"
            "   [^i]: Title — URL (site), date\n"
            f"5. Be concise yet thorough. Target length: {answer_tokens} tokens.\n"
        )
        
        # Add style profile if provided
        if style_profile_id:
            if style_profile_id == "academic":
                instructions += "\nStyle: Academic. Formal, precise language with scholarly tone. Use technical terminology where appropriate.\n"
            elif style_profile_id == "simple":
                instructions += "\nStyle: Simple. Clear, straightforward language accessible to non-experts. Avoid jargon.\n"
            elif style_profile_id == "journalist":
                instructions += "\nStyle: Journalistic. Informative, engaging, with clear explanations. Prioritize key facts.\n"
            elif style_profile_id == "technical":
                instructions += "\nStyle: Technical. Detailed, precise, assuming domain knowledge. Include technical specifics.\n"
            else:
                instructions += f"\nStyle Profile: {style_profile_id}\n"
        
        # Combine everything
        return f"{instructions}\n\n{evidence_block}"
    
    def _extract_citation_ids(self, text: str) -> Set[int]:
        """
        Extract citation IDs from text with [^i] markers.
        
        Args:
            text: Text with citation markers
            
        Returns:
            Set of citation IDs
        """
        # Find all [^i] citations
        citation_markers = re.findall(r'\[\^(\d+)\]', text)
        
        # Convert to integers and return as a set
        return {int(id) for id in citation_markers if id.isdigit()}
    
    async def generate_answer(self, query: str, evidence_list: List[RankedEvidence],
                            answer_tokens: int = 800, 
                            style_profile_id: Optional[str] = None) -> SynthesisResult:
        """
        Generate an answer from evidence with citation markers.
        
        Args:
            query: The query to answer
            evidence_list: List of ranked evidence
            answer_tokens: Target token count for the answer
            style_profile_id: Optional style profile ID
            
        Returns:
            SynthesisResult with answer and metadata
        """
        if not query or not evidence_list:
            return SynthesisResult(
                answer_markdown="No evidence available to answer this question.",
                used_citation_ids=set()
            )
        
        # Build prompts
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(query, evidence_list, answer_tokens, style_profile_id)
        
        start_time = time.time()
        try:
            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=answer_tokens,
            )
            
            # Extract answer
            answer = response.choices[0].message.content
            
            # Calculate token usage
            prompt_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(response.usage, "completion_tokens", 0) or 0

            # Extract citation IDs
            used_citation_ids = self._extract_citation_ids(answer)

            # Calculate generation time
            generation_ms = int((time.time() - start_time) * 1000)

            logger.info(
                "Generated answer in %dms, %d tokens, %d citations",
                generation_ms, completion_tokens, len(used_citation_ids)
            )

            return SynthesisResult(
                answer_markdown=answer,
                used_citation_ids=used_citation_ids,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                generation_ms=generation_ms
            )

        except Exception:
            logger.exception("Error generating answer")

            # Return a fallback result
            fallback_answer = (
                "I encountered an internal error while generating the answer. "
                "Please try again or contact support if the problem persists."
            )

            return SynthesisResult(
                answer_markdown=fallback_answer,
                used_citation_ids=set(),
                generation_ms=int((time.time() - start_time) * 1000)
            )