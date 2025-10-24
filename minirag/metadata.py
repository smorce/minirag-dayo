from typing import Any, Mapping


def metadata_matches(metadata: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
    for key, expected_value in filters.items():
        if key not in metadata:
            return False
        if str(metadata.get(key)) != str(expected_value):
            return False
    return True
