import pytest
from fastapi import HTTPException

from .character_datasets import _validate_repo_id


def test_validates_huggingface_dataset_name():
    assert _validate_repo_id("owner/my-dataset") == "owner/my-dataset"


def test_rejects_non_dataset_url():
    with pytest.raises(HTTPException):
        _validate_repo_id("https://huggingface.co/datasets/owner/name")
