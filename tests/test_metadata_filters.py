from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import asyncio
import types
from unittest.mock import patch

if "sentence_transformers" not in sys.modules:
    stub_module = types.ModuleType("sentence_transformers")
    stub_module.SentenceTransformer = object
    sys.modules["sentence_transformers"] = stub_module

from minirag.metadata import metadata_matches
from minirag.base import QueryParam
from minirag import operate as operate_module
from minirag.operate import naive_query, _find_most_related_text_unit_from_entities
from minirag.prompt import GRAPH_FIELD_SEP


class _FakeVectorDB:
    def __init__(self, results):
        self._results = results
        self.received_filters = []

    async def query(self, query, top_k, metadata_filters=None):
        self.received_filters.append(metadata_filters)
        return self._results


class _FakeTextChunksDB:
    def __init__(self, data):
        self._data = data

    async def get_by_id(self, chunk_id):
        return self._data.get(chunk_id)

    async def get_by_ids(self, chunk_ids, fields=None):
        return [self._data.get(cid) for cid in chunk_ids]


class _FakeGraphStorage:
    def __init__(self, node_sources):
        self._node_sources = node_sources

    async def get_node_edges(self, entity_name):
        return []

    async def get_node(self, entity_name):
        source_id = self._node_sources.get(entity_name, "")
        return {"source_id": source_id}


def test_metadata_matches_success():
    metadata = {"full_doc_id": "doc-123", "chunk_order_index": 2}
    filters = {"full_doc_id": "doc-123"}
    assert metadata_matches(metadata, filters)


def test_metadata_matches_value_conversion():
    metadata = {"chunk_order_index": 3}
    filters = {"chunk_order_index": "3"}
    assert metadata_matches(metadata, filters)


def test_metadata_matches_missing_key():
    metadata = {"full_doc_id": "doc-456"}
    filters = {"chunk_order_index": 1}
    assert not metadata_matches(metadata, filters)


def test_metadata_matches_value_mismatch():
    metadata = {"full_doc_id": "doc-789"}
    filters = {"full_doc_id": "doc-000"}
    assert not metadata_matches(metadata, filters)


def test_naive_query_applies_metadata_filters():
    vector_results = [{"id": "chunk-1"}, {"id": "chunk-2"}]
    vector_db = _FakeVectorDB(vector_results)
    text_chunks = _FakeTextChunksDB(
        {
            "chunk-1": {
                "content": "Doc 1",
                "full_doc_id": "doc-keep",
                "chunk_order_index": 0,
                "tokens": 10,
            },
            "chunk-2": {
                "content": "Doc 2",
                "full_doc_id": "doc-drop",
                "chunk_order_index": 1,
                "tokens": 10,
            },
        }
    )
    param = QueryParam(
        only_need_context=True,
        metadata_filters={"full_doc_id": "doc-keep"},
        top_k=2,
    )
    global_config = {"llm_model_func": lambda *args, **kwargs: ""}

    with patch.object(
        operate_module,
        "truncate_list_by_token_size",
        lambda items, **kwargs: items,
    ):
        context = asyncio.run(
            naive_query(
                "What?",
                vector_db,
                text_chunks,
                param,
                global_config,
            )
        )

    assert "Doc 1" in context
    assert "Doc 2" not in context
    assert vector_db.received_filters == [{"full_doc_id": "doc-keep"}]


def test_find_text_units_from_entities_respects_metadata_filters():
    node_datas = [
        {
            "entity_name": "A",
            "source_id": GRAPH_FIELD_SEP.join(["chunk-1"]),
            "description": "Entity A",
        },
        {
            "entity_name": "B",
            "source_id": GRAPH_FIELD_SEP.join(["chunk-2"]),
            "description": "Entity B",
        },
    ]
    text_chunks = _FakeTextChunksDB(
        {
            "chunk-1": {
                "content": "Chunk 1",
                "full_doc_id": "doc-keep",
                "chunk_order_index": 0,
                "tokens": 5,
            },
            "chunk-2": {
                "content": "Chunk 2",
                "full_doc_id": "doc-drop",
                "chunk_order_index": 1,
                "tokens": 5,
            },
        }
    )
    graph = _FakeGraphStorage(
        {
            "A": GRAPH_FIELD_SEP.join(["chunk-1"]),
            "B": GRAPH_FIELD_SEP.join(["chunk-2"]),
        }
    )
    param = QueryParam(metadata_filters={"full_doc_id": "doc-keep"})

    with patch.object(
        operate_module,
        "truncate_list_by_token_size",
        lambda items, **kwargs: items,
    ):
        text_units = asyncio.run(
            _find_most_related_text_unit_from_entities(
                node_datas,
                param,
                text_chunks,
                graph,
            )
        )

    assert len(text_units) == 1
    assert text_units[0]["full_doc_id"] == "doc-keep"


def test_build_local_query_context_filters_entities_by_metadata():
    class _EntitiesVDB:
        async def query(self, query, top_k):
            return [
                {"entity_name": "KEEP", "distance": 0.1},
                {"entity_name": "DROP", "distance": 0.2},
            ]

    class _Graph(_FakeGraphStorage):
        async def node_degree(self, entity_name):
            return 1

    graph = _Graph(
        {
            "KEEP": GRAPH_FIELD_SEP.join(["chunk-1"]),
            "DROP": GRAPH_FIELD_SEP.join(["chunk-2"]),
        }
    )
    text_chunks = _FakeTextChunksDB(
        {
            "chunk-1": {
                "content": "Chunk Keep",
                "full_doc_id": "doc-keep",
                "chunk_order_index": 0,
                "tokens": 5,
            },
            "chunk-2": {
                "content": "Chunk Drop",
                "full_doc_id": "doc-drop",
                "chunk_order_index": 1,
                "tokens": 5,
            },
        }
    )
    param = QueryParam(metadata_filters={"full_doc_id": "doc-keep"})

    with patch.object(
        operate_module,
        "truncate_list_by_token_size",
        lambda items, **kwargs: items,
    ):
        context = asyncio.run(
            operate_module._build_local_query_context(
                "query",
                graph,
                _EntitiesVDB(),
                text_chunks,
                param,
            )
        )

    assert context is not None
    assert "KEEP" in context
    assert "Chunk Keep" in context
    assert "DROP" not in context
    assert "Chunk Drop" not in context


def test_build_mini_query_context_filters_entities_by_metadata():
    class _EntityNameVDB:
        async def query(self, query, top_k):
            return [
                {"entity_name": "KEEP", "distance": 0.1},
                {"entity_name": "DROP", "distance": 0.2},
            ]

    class _RelationshipsVDB:
        async def query(self, query, top_k):
            return [{"src_id": "KEEP", "tgt_id": "DROP"}]

    class _ChunksVDB:
        async def query(self, query, top_k, metadata_filters=None):
            return [{"id": "chunk-keep"}, {"id": "chunk-drop"}]

    class _Graph(_FakeGraphStorage):
        async def get_neighbors_within_k_hops(self, key, hops):
            return {}

        async def get_node_from_types(self, type_keywords):
            return []

        async def node_degree(self, entity_name):
            return 1

    text_chunks = _FakeTextChunksDB(
        {
            "chunk-keep": {
                "content": "Chunk Keep",
                "full_doc_id": "doc-keep",
                "chunk_order_index": 0,
                "tokens": 5,
            },
            "chunk-drop": {
                "content": "Chunk Drop",
                "full_doc_id": "doc-drop",
                "chunk_order_index": 1,
                "tokens": 5,
            },
        }
    )

    async def _fake_path2chunk(*args, **kwargs):
        return {
            "KEEP": {"Score": 1, "Path": ["chunk-keep"]},
            "DROP": {"Score": 1, "Path": ["chunk-drop"]},
        }

    def _fake_cal_path_score_list(candidate, maybe):
        return {
            "KEEP": {"Score": 1, "Path": {}},
            "DROP": {"Score": 1, "Path": {}},
        }

    param = QueryParam(metadata_filters={"full_doc_id": "doc-keep"}, top_k=2)

    with patch.multiple(
        operate_module,
        truncate_list_by_token_size=lambda items, **kwargs: items,
        cal_path_score_list=_fake_cal_path_score_list,
        edge_vote_path=lambda path, edges: (path, {}),
        path2chunk=_fake_path2chunk,
        kwd2chunk=lambda ent_dict, chunk_ids, chunk_nums: chunk_ids,
    ):
        context = asyncio.run(
            operate_module._build_mini_query_context(
                ["query-entity"],
                [],
                "query",
                _Graph(
                    {
                        "KEEP": GRAPH_FIELD_SEP.join(["chunk-keep"]),
                        "DROP": GRAPH_FIELD_SEP.join(["chunk-drop"]),
                    }
                ),
                _FakeVectorDB([]),
                _EntityNameVDB(),
                _RelationshipsVDB(),
                _ChunksVDB(),
                text_chunks,
                None,
                param,
            )
        )

    assert "KEEP" in context
    assert "Chunk Keep" in context
    assert "DROP" not in context
    assert "Chunk Drop" not in context
