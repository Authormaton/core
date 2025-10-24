
"""
Service for splitting text into chunks for embedding and retrieval.
"""

from services.logging_config import get_logger
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


def _join_sentences(sentences: List[str], max_chars: int, overlap: int) -> List[str]:
    """
    Join sentences into chunks, each not exceeding max_chars in length.
    If a sentence is longer than max_chars, split it by character window with overlap.
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
            # If single sentence longer than max_chars, split by hard char window with overlap
            if sl > max_chars:
                start_idx = 0
                while start_idx < sl:
                    end_idx = min(start_idx + max_chars, sl)
                    chunks.append(s[start_idx:end_idx])
                    if end_idx == sl:
                        break
                    start_idx += max_chars - overlap
                cur = []
                cur_len = 0
            else:
                cur = [s]
                cur_len = sl
    if cur:
        chunks.append(' '.join(cur))
    return chunks


def _make_chunk_meta(text_content: str, chunk_start: int, chunk_end: int, order: int) -> Dict[str, Any]:
    """
    Create metadata dictionary for a chunk of text.
    chunk_start: The starting character index of the chunk in the original text (inclusive).
    chunk_end: The ending character index of the chunk in the original text (exclusive).
    """
    return {
        'id': uuid.uuid4().hex,
        'order': order,
        'chunk_start': chunk_start,
        'chunk_end': chunk_end,
        'text': text_content,
        'estimated_tokens': _estimate_tokens_from_text(text_content),
    }



def chunk_text(text: str, max_length: int = 500, overlap: int = 50, by_sentence: bool = True,
               min_chunk_length: int = 20, token_target: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Splits text into chunks and returns a list of chunk metadata dicts.

    Features:
    - by_sentence: attempt to split on sentence boundaries and join sentences into chunks
    - overlap: character overlap between chunks. This is the number of characters that will be shared between consecutive chunks.
    - min_chunk_length: small chunks shorter than this may be merged post-hoc
    - token_target: if provided, attempt to keep estimated token counts near this target

    Each chunk dict contains: id, order, offset, length, text, estimated_tokens
    """
    logger = get_logger(__name__)
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

        initial_chunks: List[Tuple[int, int, str]] = [] # (chunk_start, chunk_end, text_content)

        if by_sentence:
            sentences = _SENTENCE_SPLIT_RE.split(text)
            joined_chunks_str = _join_sentences(sentences, max_length, overlap)

            current_global_char_offset = 0
            for jc_str in joined_chunks_str:
                # Find the actual start of the chunk in the original text.
                # This is more robust than text.find as it tracks the global offset.
                start_idx = current_global_char_offset
                # Adjust start_idx to align with the actual content of jc_str within the remaining text
                # This handles cases where _join_sentences might have added/removed spaces or
                # if the first sentence of a chunk doesn't perfectly align with current_global_char_offset
                # due to previous overlaps.
                # We search for the jc_str in the text starting from current_global_char_offset
                # to ensure we get the correct global start index.
                found_idx = text.find(jc_str, current_global_char_offset)
                if found_idx != -1:
                    start_idx = found_idx
                else:
                    # Fallback: if not found, assume it starts at current_global_char_offset
                    # and log a warning. This should ideally not happen with _join_sentences.
                    logger.warning(f"Could not find chunk string '{jc_str[:50]}...' at or after offset {current_global_char_offset}. Assuming start at offset.")

                end_idx = start_idx + len(jc_str)
                initial_chunks.append((start_idx, end_idx, jc_str))

                # Advance the global character offset for the next chunk.
                # The next chunk's content will start after the non-overlapping part of the current chunk.
                current_global_char_offset = end_idx - overlap
                if current_global_char_offset < 0:
                    current_global_char_offset = 0

        else:
            # naive fixed-window chunking
            start_idx = 0
            while start_idx < len(text):
                end_idx = min(start_idx + max_length, len(text))
                initial_chunks.append((start_idx, end_idx, text[start_idx:end_idx]))
                if end_idx == len(text):
                    break
                start_idx += max_length - overlap

        # Optionally refine by token target (split large chunks further)
        refined_chunks: List[Tuple[int, int, str]] = []  # list of (chunk_start, chunk_end, text_content)
        for original_start, original_end, chunk_text_content in initial_chunks:
            if token_target is not None:
                est = _estimate_tokens_from_text(chunk_text_content)
                if est > token_target * 2:
                    # split by character windows approximating tokens
                    approx_chars = max(100, token_target * 4)
                    split_start = 0
                    while split_start < len(chunk_text_content):
                        split_end = min(split_start + approx_chars, len(chunk_text_content))
                        sub_chunk_text = chunk_text_content[split_start:split_end]
                        sub_chunk_global_start = original_start + split_start
                        sub_chunk_global_end = original_start + split_end
                        refined_chunks.append((sub_chunk_global_start, sub_chunk_global_end, sub_chunk_text))
                        split_start += approx_chars - overlap # Apply overlap to sub-chunks as well
                        if split_start < 0: # Ensure split_start doesn't go negative
                            split_start = 0
                        if split_end == len(chunk_text_content):
                            break
                    continue
            refined_chunks.append((original_start, original_end, chunk_text_content))

        # Convert to metadata dicts and merge tiny chunks
        chunks_meta: List[Dict[str, Any]] = []
        for i, (c_start, c_end, chunk_text_content) in enumerate(refined_chunks):
            chunks_meta.append(_make_chunk_meta(chunk_text_content, c_start, c_end, i))

        # Merge small chunks into previous chunk where appropriate
        merged: List[Dict[str, Any]] = []
        for c in chunks_meta:
            if merged and c['chunk_end'] - c['chunk_start'] < min_chunk_length:
                prev = merged[-1]
                # merge into prev
                combined_text = prev['text'] + ' ' + c['text']
                prev.update({
                    'text': combined_text,
                    'chunk_end': c['chunk_end'], # Update end to the end of the merged chunk
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
