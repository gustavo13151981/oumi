import json
from pathlib import Path
from typing import Final
from unittest.mock import patch

import pytest
from aioresponses import aioresponses

from oumi.core.configs import (
    RemoteParams,
)
from oumi.core.types.conversation import Conversation, Message, Role, Type
from oumi.inference import RemoteInferenceEngine
from oumi.inference.remote.batch_mixin import BatchStatus
from oumi.utils.io_utils import get_oumi_root_directory

_TARGET_SERVER: Final[str] = "http://fakeurl"
_TEST_IMAGE_DIR: Final[Path] = (
    get_oumi_root_directory().parent.parent.resolve() / "tests" / "testdata" / "images"
)


@pytest.mark.asyncio
async def test_upload_batch_file():
    """Test uploading a batch file."""
    with aioresponses() as m:
        m.post(
            f"{_TARGET_SERVER}/files",
            status=200,
            payload={"id": "file-123"},
        )

        engine = RemoteInferenceEngine(
            _get_default_model_params(),
            remote_params=RemoteParams(api_url=_TARGET_SERVER),
        )

        batch_requests = [
            {
                "custom_id": "request-1",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"messages": [{"role": "user", "content": "Hello"}]},
            }
        ]

        # Patch aiofiles.open to return our async context manager
        with patch("aiofiles.open", return_value=AsyncContextManagerMock()):
            file_id = await engine._upload_batch_file(batch_requests)
            assert file_id == "file-123"


@pytest.mark.asyncio
async def test_create_batch():
    """Test creating a batch job."""
    with aioresponses() as m:
        # Mock file upload
        m.post(
            f"{_TARGET_SERVER}/files",
            status=200,
            payload={"id": "file-123"},
        )
        # Mock batch creation
        m.post(
            f"{_TARGET_SERVER}/batches",
            status=200,
            payload={"id": "batch-456"},
        )

        engine = RemoteInferenceEngine(
            _get_default_model_params(),
            remote_params=RemoteParams(api_url=_TARGET_SERVER),
        )

        conversation = Conversation(
            messages=[
                Message(content="Hello", role=Role.USER, type=Type.TEXT),
            ]
        )

        with patch("aiofiles.open", return_value=AsyncContextManagerMock()):
            batch_id = await engine._create_batch(
                [conversation],
                _get_default_inference_config().generation,
            )
            assert batch_id == "batch-456"


@pytest.mark.asyncio
async def test_get_batch_status():
    """Test getting batch status."""
    with aioresponses() as m:
        m.get(
            f"{_TARGET_SERVER}/batches/batch-123",
            status=200,
            payload={
                "id": "batch-123",
                "status": "completed",
                "request_counts": {
                    "total": 10,
                    "completed": 8,
                    "failed": 1,
                },
            },
        )

        engine = RemoteInferenceEngine(
            _get_default_model_params(),
            remote_params=RemoteParams(api_url=_TARGET_SERVER),
        )

        status = await engine._get_batch_status("batch-123")
        assert status.id == "batch-123"
        assert status.status == BatchStatus.COMPLETED
        assert status.total_requests == 10
        assert status.completed_requests == 8
        assert status.failed_requests == 1


@pytest.mark.asyncio
async def test_get_batch_results():
    """Test getting batch results."""
    with aioresponses() as m:
        # Mock batch status request
        m.get(
            f"{_TARGET_SERVER}/batches/batch-123",
            status=200,
            payload={
                "id": "batch-123",
                "status": "completed",
                "request_counts": {
                    "total": 1,
                    "completed": 1,
                    "failed": 0,
                },
                "output_file_id": "file-output-123",
            },
        )

        # Mock file content request
        m.get(
            f"{_TARGET_SERVER}/files/file-output-123/content",
            status=200,
            body=json.dumps(
                {
                    "custom_id": "request-1",
                    "response": {
                        "body": {
                            "choices": [
                                {
                                    "message": {
                                        "role": "assistant",
                                        "content": "Hello there!",
                                    }
                                }
                            ]
                        }
                    },
                }
            ),
        )

        engine = RemoteInferenceEngine(
            _get_default_model_params(),
            remote_params=RemoteParams(api_url=_TARGET_SERVER),
        )

        conversation = Conversation(
            messages=[
                Message(content="Hello", role=Role.USER, type=Type.TEXT),
            ]
        )

        results = await engine._get_batch_results_with_mapping(
            "batch-123",
            [conversation],
        )

        assert len(results) == 1
        assert results[0].messages[-1].content == "Hello there!"
        assert results[0].messages[-1].role == Role.ASSISTANT


def test_infer_batch():
    """Test the public infer_batch method."""
    with aioresponses() as m:
        # Mock file upload
        m.post(
            f"{_TARGET_SERVER}/files",
            status=200,
            payload={"id": "file-123"},
        )
        # Mock batch creation
        m.post(
            f"{_TARGET_SERVER}/batches",
            status=200,
            payload={"id": "batch-456"},
        )

        engine = RemoteInferenceEngine(
            _get_default_model_params(),
            remote_params=RemoteParams(api_url=_TARGET_SERVER),
        )

        conversation = Conversation(
            messages=[
                Message(content="Hello", role=Role.USER, type=Type.TEXT),
            ]
        )

        # Use AsyncContextManagerMock instead of AsyncMock
        with patch("aiofiles.open", return_value=AsyncContextManagerMock()):
            batch_id = engine.infer_batch(
                [conversation], _get_default_inference_config()
            )
            assert batch_id == "batch-456"


def test_get_batch_status_public():
    """Test the public get_batch_status method."""
    with aioresponses() as m:
        m.get(
            f"{_TARGET_SERVER}/batches/batch-123",
            status=200,
            payload={
                "id": "batch-123",
                "status": "in_progress",
                "request_counts": {
                    "total": 10,
                    "completed": 5,
                    "failed": 0,
                },
            },
        )

        engine = RemoteInferenceEngine(
            _get_default_model_params(),
            remote_params=RemoteParams(api_url=_TARGET_SERVER),
        )

        status = engine.get_batch_status("batch-123")
        assert status.id == "batch-123"
        assert status.status == BatchStatus.IN_PROGRESS
        assert status.total_requests == 10
        assert status.completed_requests == 5
        assert status.failed_requests == 0


def test_get_batch_results_public():
    """Test the public get_batch_results method."""
    with aioresponses() as m:
        # Mock batch status request
        m.get(
            f"{_TARGET_SERVER}/batches/batch-123",
            status=200,
            payload={
                "id": "batch-123",
                "status": "completed",
                "request_counts": {
                    "total": 1,
                    "completed": 1,
                    "failed": 0,
                },
                "output_file_id": "file-output-123",
            },
        )

        # Mock file content request
        m.get(
            f"{_TARGET_SERVER}/files/file-output-123/content",
            status=200,
            body=json.dumps(
                {
                    "custom_id": "request-1",
                    "response": {
                        "body": {
                            "choices": [
                                {
                                    "message": {
                                        "role": "assistant",
                                        "content": "Hello there!",
                                    }
                                }
                            ]
                        }
                    },
                }
            ),
        )

        engine = RemoteInferenceEngine(
            _get_default_model_params(),
            remote_params=RemoteParams(api_url=_TARGET_SERVER),
        )

        conversation = Conversation(
            messages=[
                Message(content="Hello", role=Role.USER, type=Type.TEXT),
            ]
        )

        results = engine.get_batch_results(
            "batch-123",
            [conversation],
        )

        assert len(results) == 1
        assert results[0].messages[-1].content == "Hello there!"
        assert results[0].messages[-1].role == Role.ASSISTANT


@pytest.mark.asyncio
async def test_list_batches():
    """Test listing batch jobs."""
    with aioresponses() as m:
        # Mock with exact URL including query parameters
        m.get(
            f"{_TARGET_SERVER}/batches?limit=1",  # Include query params in URL
            status=200,
            payload={
                "object": "list",
                "data": [
                    {
                        "id": "batch_abc123",
                        "object": "batch",
                        "endpoint": "/v1/chat/completions",
                        "input_file_id": "file-abc123",
                        "batch_completion_window": "24h",
                        "status": "completed",
                        "output_file_id": "file-cvaTdG",
                        "error_file_id": "file-HOWS94",
                        "created_at": 1711471533,
                        "request_counts": {"total": 100, "completed": 95, "failed": 5},
                        "metadata": {"batch_description": "Test batch"},
                    }
                ],
                "first_id": "batch_abc123",
                "last_id": "batch_abc123",
                "has_more": False,
            },
        )

        engine = RemoteInferenceEngine(
            _get_default_model_params(),
            remote_params=RemoteParams(api_url=_TARGET_SERVER),
        )

        response = await engine._list_batches(limit=1)

        assert len(response.batches) == 1
        batch = response.batches[0]
        assert batch.id == "batch_abc123"
        assert batch.status == BatchStatus.COMPLETED
        assert batch.total_requests == 100
        assert batch.completed_requests == 95
        assert batch.failed_requests == 5
        assert batch.metadata == {"batch_description": "Test batch"}
        assert response.first_id == "batch_abc123"
        assert response.last_id == "batch_abc123"
        assert not response.has_more


def test_list_batches_public():
    """Test the public list_batches method."""
    with aioresponses() as m:
        m.get(
            f"{_TARGET_SERVER}/batches?limit=2",  # Include query params in URL
            status=200,
            payload={
                "object": "list",
                "data": [
                    {
                        "id": "batch_1",
                        "endpoint": "/v1/chat/completions",
                        "status": "completed",
                        "input_file_id": "file-1",
                        "batch_completion_window": "24h",
                    },
                    {
                        "id": "batch_2",
                        "endpoint": "/v1/chat/completions",
                        "status": "in_progress",
                        "input_file_id": "file-2",
                        "batch_completion_window": "24h",
                    },
                ],
                "first_id": "batch_1",
                "last_id": "batch_2",
                "has_more": True,
            },
        )

        engine = RemoteInferenceEngine(
            _get_default_model_params(),
            remote_params=RemoteParams(api_url=_TARGET_SERVER),
        )

        response = engine.list_batches(limit=2)

        assert len(response.batches) == 2
        assert response.first_id == "batch_1"
        assert response.last_id == "batch_2"
        assert response.has_more
