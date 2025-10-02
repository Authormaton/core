
"""
Service for splitting text into chunks for embedding and retrieval.
"""

import logging
import uuid
import re
from typing import List, Dict, Any, Optional, Tuple
from services.exceptions import DocumentChunkError

# Try to import tiktoken for more accurate token counting; fall back to simple estimator
try:
    import tiktoken  # type: ignore
except Exception:
    tiktoken = None


_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')


def _estimate_tokens_from_text(text: str) -> int:
    """
    Estimate the number of tokens in the given text.
    Uses tiktoken if available, otherwise falls back to a naive estimator.
    """
    if tiktoken is not None:
        try:
            enc = tiktoken.get_encoding('gpt2')
            return len(enc.encode(text))
        except Exception:
            pass
    # fallback naive estimator: 1 token ≈ 4 characters
    return max(1, len(text) // 4)


def _join_sentences(sentences: List[str], max_chars: int) -> List[str]:
    """
    Join sentences into chunks, each not exceeding max_chars in length.
    If a sentence is longer than max_chars, split it by character window.
    """
    chunks = []
    cur = []
    cur_len = 0
    for s in sentences:
        sl = len(s)
        if cur_len + sl + (1 if cur else 0) <= max_chars:
            cur.append(s)
            cur_len += sl + (1 if cur else 0)
        else:
            if cur:
                chunks.append(' '.join(cur))
            # If single sentence longer than max_chars, split by hard char window
            if sl > max_chars:
                for i in range(0, sl, max_chars):
                    chunks.append(s[i:i+max_chars])
                cur = []
                cur_len = 0
            else:
                cur = [s]
                cur_len = sl
    if cur:
        chunks.append(' '.join(cur))
    return chunks


def _make_chunk_meta(text: str, offset: int, order: int) -> Dict[str, Any]:
    """
    Create metadata dictionary for a chunk of text.
    """
    return {
        'id': uuid.uuid4().hex,
        'order': order,
        'offset': offset,
        'length': len(text),
        'text': text,
        'estimated_tokens': _estimate_tokens_from_text(text),
    }



def chunk_text(text: str, max_length: int = 500, overlap: int = 50, by_sentence: bool = True,
               min_chunk_length: int = 20, token_target: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Splits text into chunks and returns a list of chunk metadata dicts.

    Features:
    - by_sentence: attempt to split on sentence boundaries and join sentences into chunks
    - overlap: character overlap between chunks
    - min_chunk_length: small chunks shorter than this may be merged post-hoc
    - token_target: if provided, attempt to keep estimated token counts near this target

    Each chunk dict contains: id, order, offset, length, text, estimated_tokens
    """
    logger = logging.getLogger(__name__)
    try:
        if not text:
            return []
        if max_length <= 0:
            logger.error("Invalid max_length for chunking: %d", max_length)
            raise DocumentChunkError("max_length must be greater than 0.")
        if overlap < 0:
            logger.error("Invalid overlap for chunking: %d", overlap)
            raise DocumentChunkError("overlap must be >= 0.")
        if overlap >= max_length:
            logger.error("overlap >= max_length for chunking: %d >= %d", overlap, max_length)
            raise DocumentChunkError("overlap must be less than max_length.")

        # Sentence-aware splitting
        if by_sentence:
            sentences = _SENTENCE_SPLIT_RE.split(text)
            # join sentences into approximate max_length chunks
            raw_chunks = _join_sentences(sentences, max_length)
        else:
            # naive fixed-window chunking
            raw_chunks = []
            start = 0
            while start < len(text):
                end = min(start + max_length, len(text))
                raw_chunks.append(text[start:end])
                if end == len(text):
                    break
                start += max_length - overlap

        # Optionally refine by token target (split large chunks further)
        refined: List[Tuple[int, str]] = []  # list of (offset, text)
        offset = 0
        for rc in raw_chunks:
            off = text.find(rc, offset)
            if off == -1:
                off = offset
            # if token_target set and estimated tokens exceed twice the target, split
            if token_target is not None:
                est = _estimate_tokens_from_text(rc)
                if est > token_target * 2:
                    # split by character windows approximating tokens
                    approx_chars = max(100, token_target * 4)
                    for i in range(0, len(rc), approx_chars):
                        refined.append((off + i, rc[i:i+approx_chars]))
                    offset = off + len(rc)
                    continue
            refined.append((off, rc))
            offset = off + len(rc)

        # Convert to metadata dicts and merge tiny chunks
        chunks_meta: List[Dict[str, Any]] = []
        for i, (off, chunk_text_content) in enumerate(refined):
            chunks_meta.append(_make_chunk_meta(chunk_text_content, off, i))

        # Merge small chunks into previous chunk where appropriate
        merged: List[Dict[str, Any]] = []
        for c in chunks_meta:
            if merged and c['length'] < min_chunk_length:
                prev = merged[-1]
                # merge into prev
                combined_text = prev['text'] + ' ' + c['text']
                prev.update({
                    'text': combined_text,
                    'length': len(combined_text),
                    'estimated_tokens': _estimate_tokens_from_text(combined_text),
                })
            else:
                merged.append(c)

        # Re-assign order and ensure offsets are correct
        for idx, c in enumerate(merged):
            c['order'] = idx
        return merged
    except Exception as e:
        logger.exception("Error during text chunking")
        raise DocumentChunkError(f"Failed to chunk text: {e}") from e
