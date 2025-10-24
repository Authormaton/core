
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
            # New regex to match sentences including trailing punctuation and whitespace
            _SENTENCE_MATCH_RE = re.compile(r'[^.!?]*[.!?]+(?=\s*|$)', re.DOTALL)
            sentences_with_spans: List[Tuple[int, int, str]] = []
            for m in _SENTENCE_MATCH_RE.finditer(text):
                # Ensure we capture the full sentence including trailing whitespace if present
                sentence_text = m.group(0)
                sentences_with_spans.append((m.start(), m.end(), sentence_text))

            current_chunk_sentences: List[Tuple[int, int, str]] = []
            current_chunk_char_length = 0
            current_chunk_start_idx = 0

            for i, (s_start, s_end, s_text) in enumerate(sentences_with_spans):
                # If this is the first sentence in a potential chunk, set its start index
                if not current_chunk_sentences:
                    current_chunk_start_idx = s_start

                # Check if adding the current sentence exceeds max_length
                # We add 1 for a potential space if joining multiple sentences
                potential_new_length = current_chunk_char_length + len(s_text) + (1 if current_chunk_sentences else 0)

                if potential_new_length <= max_length:
                    current_chunk_sentences.append((s_start, s_end, s_text))
                    current_chunk_char_length = potential_new_length
                else:
                    # Finalize the current chunk
                    if current_chunk_sentences:
                        chunk_end_idx = current_chunk_sentences[-1][1]
                        chunk_text_content = text[current_chunk_start_idx:chunk_end_idx]
                        initial_chunks.append((current_chunk_start_idx, chunk_end_idx, chunk_text_content))

                    # Start a new chunk with the current sentence
                    current_chunk_sentences = [(s_start, s_end, s_text)]
                    current_chunk_char_length = len(s_text)
                    current_chunk_start_idx = s_start

            # Add the last chunk if any sentences are remaining
            if current_chunk_sentences:
                chunk_end_idx = current_chunk_sentences[-1][1]
                chunk_text_content = text[current_chunk_start_idx:chunk_end_idx]
                initial_chunks.append((current_chunk_start_idx, chunk_end_idx, chunk_text_content))

            # Apply overlap logic to the initial_chunks
            # Overlap is applied by adjusting the start of subsequent chunks
            # to be within the previous chunk's end, ensuring sentence boundaries are respected.
            # This means the overlap is in terms of characters, but the chunk starts at a sentence boundary.
            overlapped_chunks: List[Tuple[int, int, str]] = []
            if initial_chunks:
                overlapped_chunks.append(initial_chunks[0]) # First chunk is always added as is

                for i in range(1, len(initial_chunks)):
                    prev_chunk_start, prev_chunk_end, _ = initial_chunks[i-1]
                    current_chunk_start, current_chunk_end, _ = initial_chunks[i]

                    # Calculate the desired overlap start point
                    desired_overlap_start = prev_chunk_end - overlap

                    # Find the sentence that starts at or after desired_overlap_start
                    # and is before or at the current_chunk_start
                    new_chunk_start_idx = current_chunk_start # Default to current chunk start

                    for s_start, s_end, s_text in sentences_with_spans:
                        if s_start >= desired_overlap_start and s_start < current_chunk_start:
                            new_chunk_start_idx = s_start
                            break
                        elif s_start >= current_chunk_start: # If we passed the current chunk start, stop
                            break

                    # Ensure the new chunk start is not greater than the current chunk's original start
                    new_chunk_start_idx = min(new_chunk_start_idx, current_chunk_start)
                    # Ensure the new chunk start is not less than 0
                    new_chunk_start_idx = max(0, new_chunk_start_idx)

                    # Reconstruct the chunk text based on the new start and original end
                    chunk_text_content = text[new_chunk_start_idx:current_chunk_end]
                    overlapped_chunks.append((new_chunk_start_idx, current_chunk_end, chunk_text_content))
            initial_chunks = overlapped_chunks

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
