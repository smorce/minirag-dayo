from typing import Any, Mapping


def metadata_matches(data: Mapping[str, Any], filters: Mapping[str, Any]) -> bool:
    """
    Checks if the given data object matches the metadata filters.
    The metadata can be at the top level of the data object or nested under a "metadata" key.
    """
    metadata_to_check = data
    if "metadata" in data and isinstance(data["metadata"], Mapping):
        metadata_to_check = data["metadata"]

    for key, expected_value in filters.items():
        if key not in metadata_to_check:
            return False
        if str(metadata_to_check.get(key)) != str(expected_value):
            return False
    return True
