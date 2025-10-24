import asyncio
import os
import shutil
import unittest
from minirag import MiniRAG, QueryParam
from minirag.utils import EmbeddingFunc

# Mock embedding function for testing
async def mock_embedding_func(texts: list[str]) -> list[list[float]]:
    # Simple mock embedding function that returns a fixed-size vector of 0.1s
    return [[0.1] * 384 for _ in texts]

# Mock LLM function for testing
async def mock_llm_func(prompt: str, **kwargs) -> str:
    return f"Mock response to: {prompt}"

class TestMetadataFilter(unittest.TestCase):
    def setUp(self):
        # Create a temporary working directory for each test
        self.working_dir = "test_metadata_working_dir"
        os.makedirs(self.working_dir, exist_ok=True)
        self.rag = MiniRAG(
            working_dir=self.working_dir,
            embedding_func=EmbeddingFunc(
                embedding_dim=384,
                max_token_size=1000,
                func=mock_embedding_func,
            ),
            llm_model_func=mock_llm_func,
        )
        # Clear the document status before each test
        asyncio.run(self.rag.doc_status.drop())

    def tearDown(self):
        # Clean up the temporary working directory after each test
        shutil.rmtree(self.working_dir)

    def test_metadata_insertion_and_filtering(self):
        # Test case for inserting documents with metadata and filtering them
        sample_texts = [
            "Today was a wonderful day. I took a walk in the park.",
            "Yesterday, I finished a major project at work.",
            "This weekend, I tried cooking a new pasta dish.",
            "I enjoy reading, and I'm currently reading 'Norwegian Wood'.",
        ]
        sample_metadatas = [
            {"source": "personal_journal", "category": "leisure"},
            {"source": "work_log", "category": "professional"},
            {"source": "personal_journal", "category": "hobby"},
            {"source": "reading_list", "category": "leisure"},
        ]

        asyncio.run(self.rag.ainsert(sample_texts, metadatas=sample_metadatas))

        # Test filtering by a single metadata field
        query = "What did I do for leisure?"
        param = QueryParam(mode="naive", metadata_filters={"category": "leisure"}, only_need_context=True)
        result = asyncio.run(self.rag.aquery(query, param=param))
        self.assertIn("park", result.lower())
        self.assertIn("wood", result.lower())
        self.assertNotIn("project", result.lower())

        # Test filtering by multiple metadata fields
        param = QueryParam(
            mode="naive",
            metadata_filters={"source": "personal_journal", "category": "leisure"},
            only_need_context=True,
        )
        result = asyncio.run(self.rag.aquery(query, param=param))
        self.assertIn("park", result.lower())
        self.assertNotIn("wood", result.lower())

        # Test with no matching metadata
        param = QueryParam(mode="naive", metadata_filters={"category": "non_existent"}, only_need_context=True)
        result = asyncio.run(self.rag.aquery(query, param=param))
        self.assertEqual("", result)

    def test_insertion_without_metadata(self):
        # Test case for inserting documents without metadata
        sample_texts = [
            "This is a document without metadata.",
            "This is another document without metadata.",
        ]
        asyncio.run(self.rag.ainsert(sample_texts))

        query = "Tell me about the documents."
        param = QueryParam(mode="naive", only_need_context=True)
        result = asyncio.run(self.rag.aquery(query, param=param))
        self.assertIn("document", result.lower())

        # Test that filtering with a metadata key returns nothing
        param = QueryParam(mode="naive", metadata_filters={"source": "any"}, only_need_context=True)
        result = asyncio.run(self.rag.aquery(query, param=param))
        self.assertEqual("", result)

    def test_mixed_metadata_insertion(self):
        # Test case for inserting documents with and without metadata
        sample_texts = [
            "Document with metadata.",
            "Document without metadata.",
        ]
        sample_metadatas = [
            {"source": "mixed_test"},
            {},
        ]
        # The ainsert function should handle a mix of metadata and empty dict
        asyncio.run(self.rag.ainsert(sample_texts, metadatas=sample_metadatas))

        # Query for the document with metadata
        query = "Tell me about the documents."
        param = QueryParam(mode="naive", metadata_filters={"source": "mixed_test"}, only_need_context=True)
        result = asyncio.run(self.rag.aquery(query, param=param))
        self.assertIn("metadata", result.lower())

        # Query without filters should return both documents
        param = QueryParam(mode="naive", only_need_context=True)
        result = asyncio.run(self.rag.aquery(query, param=param))
        self.assertIn("with metadata", result.lower())
        self.assertIn("without metadata", result.lower())

if __name__ == "__main__":
    unittest.main()
