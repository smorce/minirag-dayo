from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import importlib.util

MODULE_PATH = PROJECT_ROOT / "minirag" / "metadata.py"
spec = importlib.util.spec_from_file_location("metadata_module", MODULE_PATH)
metadata_module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(metadata_module)
metadata_matches = metadata_module.metadata_matches


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
