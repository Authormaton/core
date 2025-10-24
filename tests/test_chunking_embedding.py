import sys
import os
import pytest
from dotenv import load_dotenv
from services.chunking_service import chunk_text
from services.embedding_service import embed_texts

load_dotenv()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
def test_chunk_text_basic():
    text = "abcdefghijklmnopqrstuvwxyz"
    max_length = 10
    overlap = 2
    chunks = chunk_text(text, max_length=max_length, overlap=overlap, by_sentence=False)

    assert len(chunks) > 1
    for i, chunk in enumerate(chunks):
        assert "id" in chunk
        assert "order" in chunk
        assert "chunk_start" in chunk
        assert "chunk_end" in chunk
        assert "text" in chunk
        assert "estimated_tokens" in chunk

        assert chunk["order"] == i
        assert len(chunk["text"]) <= max_length
        assert chunk["chunk_end"] - chunk["chunk_start"] == len(chunk["text"])
        assert text[chunk["chunk_start"] : chunk["chunk_end"]] == chunk["text"]

    # Check overlap for consecutive chunks
    if len(chunks) > 1:
        for i in range(len(chunks) - 1):
            current_chunk = chunks[i]
            next_chunk = chunks[i+1]
            # The end of the current chunk minus the overlap should be the start of the next chunk
            expected_next_start = current_chunk["chunk_start"] + max_length - overlap
            # For fixed-window chunking, the next chunk's start should be exactly as expected
            assert next_chunk["chunk_start"] == expected_next_start


def test_chunk_text_with_metadata_and_overlap():
    long_text = "This is the first sentence. This is the second sentence, which is quite a bit longer than the first one and will likely be split. This is the third sentence. And finally, the fourth sentence to test the end boundary conditions."
    max_length = 50
    overlap = 10
    chunks = chunk_text(long_text, max_length=max_length, overlap=overlap, by_sentence=True)

    assert len(chunks) > 1

    for i, chunk in enumerate(chunks):
        assert "id" in chunk
        assert "order" in chunk
        assert "chunk_start" in chunk
        assert "chunk_end" in chunk
        assert "text" in chunk
        assert "estimated_tokens" in chunk

        assert chunk["order"] == i
        assert len(chunk["text"]) <= max_length
        assert chunk["chunk_end"] - chunk["chunk_start"] == len(chunk["text"])
        assert long_text[chunk["chunk_start"] : chunk["chunk_end"]] == chunk["text"]

        # Verify that chunk_start and chunk_end are within the bounds of the original text
        assert 0 <= chunk["chunk_start"] < len(long_text)
        assert 0 < chunk["chunk_end"] <= len(long_text)
        assert chunk["chunk_start"] < chunk["chunk_end"]

    # Verify overlap for sentence-aware chunking with long sentences
    # This is more complex due to sentence boundaries, so we check for *some* overlap
    # and logical progression of chunk_start/chunk_end
    for i in range(len(chunks) - 1):
        current_chunk = chunks[i]
        next_chunk = chunks[i+1]

        # Ensure there is some overlap or direct succession
        assert next_chunk["chunk_start"] < current_chunk["chunk_end"]

    # Test naive fixed-window chunking with metadata
    text_naive = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    max_length_naive = 10
    overlap_naive = 3
    chunks_naive = chunk_text(text_naive, max_length=max_length_naive, overlap=overlap_naive, by_sentence=False)

    assert len(chunks_naive) > 1
    for i, chunk in enumerate(chunks_naive):
        assert chunk["order"] == i
        assert len(chunk["text"]) <= max_length_naive
        assert chunk["chunk_end"] - chunk["chunk_start"] == len(chunk["text"])
        assert text_naive[chunk["chunk_start"] : chunk["chunk_end"]] == chunk["text"]

    # Check exact overlap for naive fixed-window chunking
    for i in range(len(chunks_naive) - 1):
        current_chunk = chunks_naive[i]
        next_chunk = chunks_naive[i+1]
        expected_next_start = current_chunk["chunk_start"] + max_length_naive - overlap_naive
        assert next_chunk["chunk_start"] == expected_next_start

    # Test with min_chunk_length merging
    short_text = "Sentence one. Two. Three. Four. Five."
    chunks_merged = chunk_text(short_text, max_length=20, overlap=0, min_chunk_length=5)
    # Expect some chunks to be merged, so fewer chunks than sentences
    assert len(chunks_merged) < len(short_text.split(". "))
    for chunk in chunks_merged:
        assert chunk["chunk_end"] - chunk["chunk_start"] == len(chunk["text"])
        assert short_text[chunk["chunk_start"] : chunk["chunk_end"]] == chunk["text"]

    # Test with token_target splitting
    very_long_sentence = "A very long sentence that definitely exceeds the token target and should be split into multiple sub-chunks. " * 10
    chunks_token_split = chunk_text(very_long_sentence, max_length=100, overlap=10, token_target=20)
    assert len(chunks_token_split) > 1
    for chunk in chunks_token_split:
        assert chunk["chunk_end"] - chunk["chunk_start"] == len(chunk["text"])
        assert very_long_sentence[chunk["chunk_start"] : chunk["chunk_end"]] == chunk["text"]


def test_chunk_text_boundary_overlap_and_metadata():
    # Test case 1: Fixed-window chunking with significant overlap
    text1 = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    max_length1 = 10
    overlap1 = 5
    chunks1 = chunk_text(text1, max_length=max_length1, overlap=overlap1, by_sentence=False)

    expected_chunks1 = [
        (0, 10, "abcdefghij"),
        (5, 15, "fghijklmno"),
        (10, 20, "klmnopqrst"),
        (15, 25, "pqrstuvwxy"),
        (20, 30, "uvwxyzABCD"),
        (25, 35, "zABCDEFGHI"),
        (30, 40, "FGHIJKLMNO"),
        (35, 45, "LMNOPQRSTU"),
        (40, 50, "UVWXYZ"),
        (45, 52, "TUVWXYZ")
    ]

    assert len(chunks1) == len(expected_chunks1)
    for i, chunk in enumerate(chunks1):
        assert chunk["chunk_start"] == expected_chunks1[i][0]
        assert chunk["chunk_end"] == expected_chunks1[i][1]
        assert chunk["text"] == expected_chunks1[i][2]
        assert text1[chunk["chunk_start"] : chunk["chunk_end"]] == chunk["text"]

    # Test case 2: Sentence-aware chunking with overlap and varying sentence lengths
    text2 = "This is sentence one. This is sentence two, which is a bit longer. And this is sentence three, the final one."
    max_length2 = 30
    overlap2 = 10
    chunks2 = chunk_text(text2, max_length=max_length2, overlap=overlap2, by_sentence=True)

    assert len(chunks2) > 0
    for i, chunk in enumerate(chunks2):
        assert 0 <= chunk["chunk_start"] < len(text2)
        assert 0 < chunk["chunk_end"] <= len(text2)
        assert chunk["chunk_start"] < chunk["chunk_end"]
        assert text2[chunk["chunk_start"] : chunk["chunk_end"]] == chunk["text"]
        assert len(chunk["text"]) <= max_length2

    # Verify overlap for consecutive chunks in text2
    for i in range(len(chunks2) - 1):
        current_chunk = chunks2[i]
        next_chunk = chunks2[i+1]
        # The start of the next chunk should be within the overlap region of the current chunk
        assert next_chunk["chunk_start"] < current_chunk["chunk_end"]
        # And the start of the next chunk should be after the non-overlapping part of the current chunk
        assert next_chunk["chunk_start"] >= current_chunk["chunk_end"] - overlap2

def test_chunk_text_empty():
    assert chunk_text("") == []


def test_embed_texts_shape(mocker):
    fake_response = type('FakeResponse', (), {
        'data': [
            type('FakeEmbedding', (), {'embedding': [0.1, 0.2, 0.3]})(),
            type('FakeEmbedding', (), {'embedding': [0.4, 0.5, 0.6]})()
        ]
    })()
    mocker.patch("services.embedding_service.get_openai_api_key", return_value="sk-test")
    mock_client = mocker.Mock()
    mock_client.embeddings.create.return_value = fake_response
    mocker.patch("services.embedding_service.OpenAI", return_value=mock_client)
    texts = ["hello world", "test embedding"]
    embeddings = embed_texts(texts)
    assert isinstance(embeddings, list)
    assert len(embeddings) == 2
    assert all(isinstance(vec, list) for vec in embeddings)
    assert all(isinstance(x, float) for vec in embeddings for x in vec)
    assert all(len(vec) > 0 for vec in embeddings)


def test_chunk_text_invalid_params():
    # max_length <= 0
    with pytest.raises(ValueError, match="max_length must be greater than 0"):
        chunk_text("abc", max_length=0, overlap=0)
    with pytest.raises(ValueError, match="max_length must be greater than 0"):
        chunk_text("abc", max_length=-1, overlap=0)
    # overlap < 0
    with pytest.raises(ValueError, match="overlap must be >= 0"):
        chunk_text("abc", max_length=5, overlap=-1)
    # overlap >= max_length
    with pytest.raises(ValueError, match="overlap must be less than max_length"):
        chunk_text("abc", max_length=5, overlap=5)
    with pytest.raises(ValueError, match="overlap must be less than max_length"):
        chunk_text("abc", max_length=5, overlap=6)

@pytest.mark.integration
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
def test_embed_texts_openai_live():
    texts = ["OpenAI test", "Embedding API"]
    embeddings = embed_texts(texts)
    assert isinstance(embeddings, list)
    assert len(embeddings) == 2
    assert all(isinstance(vec, list) for vec in embeddings)
    assert all(isinstance(x, float) for vec in embeddings for x in vec)
    assert all(len(vec) > 0 for vec in embeddings)
    assert all(isinstance(x, float) for vec in embeddings for x in vec)
    # Embedding size is model-dependent, but should be >0
    assert all(len(vec) > 0 for vec in embeddings)

def test_chunk_text_edge_cases():
    # Test with irregular spacing (double spaces, tabs, newlines)
    text_irregular_spacing = "Sentence one.  \tSentence two.\nSentence three."
    chunks_irregular = chunk_text(text_irregular_spacing, max_length=30, overlap=5, by_sentence=True)
    assert len(chunks_irregular) == 2
    assert chunks_irregular[0]["text"] == "Sentence one.  \tSentence two."
    assert chunks_irregular[0]["chunk_start"] == 0
    assert chunks_irregular[0]["chunk_end"] == 29
    assert chunks_irregular[1]["text"] == "Sentence two.\nSentence three."
    assert chunks_irregular[1]["chunk_start"] == 17 # Overlap starts from here
    assert chunks_irregular[1]["chunk_end"] == 47

    # Test with repeated sentence text
    text_repeated_sentences = "This is a test. This is a test. This is another test."
    chunks_repeated = chunk_text(text_repeated_sentences, max_length=20, overlap=0, by_sentence=True)
    assert len(chunks_repeated) == 3
    assert chunks_repeated[0]["text"] == "This is a test."
    assert chunks_repeated[0]["chunk_start"] == 0
    assert chunks_repeated[0]["chunk_end"] == 15
    assert chunks_repeated[1]["text"] == "This is a test."
    assert chunks_repeated[1]["chunk_start"] == 16
    assert chunks_repeated[1]["chunk_end"] == 31
    assert chunks_repeated[2]["text"] == "This is another test."
    assert chunks_repeated[2]["chunk_start"] == 32
    assert chunks_repeated[2]["chunk_end"] == 53

    # Test overlap behavior for sentence-aware chunking (more explicit check)
    text_overlap_sentence = "First sentence. Second sentence. Third sentence."
    chunks_overlap_sentence = chunk_text(text_overlap_sentence, max_length=30, overlap=10, by_sentence=True)
    assert len(chunks_overlap_sentence) == 2
    assert chunks_overlap_sentence[0]["text"] == "First sentence. Second sentence."
    assert chunks_overlap_sentence[0]["chunk_start"] == 0
    assert chunks_overlap_sentence[0]["chunk_end"] == 32
    # Expected overlap: 10 chars from "Second sentence." (length 16). So, new chunk should start at index 22.
    # "Second sentence. Third sentence."
    assert chunks_overlap_sentence[1]["text"] == "Second sentence. Third sentence."
    assert chunks_overlap_sentence[1]["chunk_start"] == 16 # Should start at the beginning of "Second sentence."
    assert chunks_overlap_sentence[1]["chunk_end"] == 50

    # Test overlap behavior for naive chunking (already covered, but adding for completeness)
    text_overlap_naive = "abcdefghijklmnopqrstuvwxyz"
    chunks_overlap_naive = chunk_text(text_overlap_naive, max_length=10, overlap=3, by_sentence=False)
    assert len(chunks_overlap_naive) == 3
    assert chunks_overlap_naive[0]["text"] == "abcdefghij"
    assert chunks_overlap_naive[0]["chunk_start"] == 0
    assert chunks_overlap_naive[0]["chunk_end"] == 10
    assert chunks_overlap_naive[1]["text"] == "hijklmnopq"
    assert chunks_overlap_naive[1]["chunk_start"] == 7 # 10 - 3 = 7
    assert chunks_overlap_naive[1]["chunk_end"] == 17
    assert chunks_overlap_naive[2]["text"] == "opqrstuvwx"
    assert chunks_overlap_naive[2]["chunk_start"] == 14 # 17 - 3 = 14
    assert chunks_overlap_naive[2]["chunk_end"] == 24

    # Test with overlap >= max_length (should raise error)
    with pytest.raises(ValueError, match="overlap must be less than max_length"):
        chunk_text("abc", max_length=5, overlap=5, by_sentence=True)
    with pytest.raises(ValueError, match="overlap must be less than max_length"):
        chunk_text("abc", max_length=5, overlap=6, by_sentence=False)

    # New tests for token_target small values with large overlap
    text_long = "A" * 200
    # Case 1: overlap >= approx_chars, should still make progress
    chunks_small_token_large_overlap = chunk_text(text_long, max_length=50, overlap=40, token_target=5, by_sentence=False)
    assert len(chunks_small_token_large_overlap) > 1
    assert chunks_small_token_large_overlap[0]["text"] == text_long[0:20]
    assert chunks_small_token_large_overlap[1]["text"] == text_long[1:21]
    assert chunks_small_token_large_overlap[-1]["chunk_end"] == len(text_long)

    # Case 2: overlap = approx_chars - 1
    chunks_overlap_minus_1 = chunk_text(text_long, max_length=50, overlap=19, token_target=5, by_sentence=False)
    assert len(chunks_overlap_minus_1) > 1
    assert chunks_overlap_minus_1[0]["text"] == text_long[0:20]
    assert chunks_overlap_minus_1[1]["text"] == text_long[1:21]
    assert chunks_overlap_minus_1[-1]["chunk_end"] == len(text_long)

    # Case 3: overlap = approx_chars
    chunks_overlap_equal = chunk_text(text_long, max_length=50, overlap=20, token_target=5, by_sentence=False)
    assert len(chunks_overlap_equal) > 1
    assert chunks_overlap_equal[0]["text"] == text_long[0:20]
    assert chunks_overlap_equal[1]["text"] == text_long[1:21]
    assert chunks_overlap_equal[-1]["chunk_end"] == len(text_long)