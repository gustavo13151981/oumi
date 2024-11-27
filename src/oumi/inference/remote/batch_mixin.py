import json
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import aiofiles
import aiohttp
import jsonlines

from oumi.core.configs import GenerationParams, RemoteParams
from oumi.core.types.conversation import Conversation


class BatchStatus(Enum):
    """Status of a batch inference job."""

    VALIDATING = "validating"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass
class BatchInfo:
    """Information about a batch job."""

    id: str
    status: BatchStatus
    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    endpoint: Optional[str] = None
    input_file_id: Optional[str] = None
    batch_completion_window: Optional[str] = None
    output_file_id: Optional[str] = None
    error_file_id: Optional[str] = None
    error: Optional[str] = None
    created_at: Optional[datetime] = None
    in_progress_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    finalizing_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    failed_at: Optional[datetime] = None
    expired_at: Optional[datetime] = None
    cancelling_at: Optional[datetime] = None
    cancelled_at: Optional[datetime] = None
    metadata: Optional[dict[str, Any]] = None

    @staticmethod
    def _convert_timestamp(timestamp: Optional[int]) -> Optional[datetime]:
        """Convert Unix timestamp to datetime."""
        return datetime.fromtimestamp(timestamp) if timestamp is not None else None

    @classmethod
    def from_api_response(cls, response: dict[str, Any]) -> "BatchInfo":
        """Create BatchInfo from API response dictionary."""
        return cls(
            id=response["id"],
            status=BatchStatus(response["status"]),
            endpoint=response.get("endpoint"),
            input_file_id=response.get("input_file_id"),
            batch_completion_window=response.get("batch_completion_window"),
            output_file_id=response.get("output_file_id"),
            error_file_id=response.get("error_file_id"),
            error=response.get("error"),
            created_at=cls._convert_timestamp(response.get("created_at")),
            in_progress_at=cls._convert_timestamp(response.get("in_progress_at")),
            expires_at=cls._convert_timestamp(response.get("expires_at")),
            finalizing_at=cls._convert_timestamp(response.get("finalizing_at")),
            completed_at=cls._convert_timestamp(response.get("completed_at")),
            failed_at=cls._convert_timestamp(response.get("failed_at")),
            expired_at=cls._convert_timestamp(response.get("expired_at")),
            cancelling_at=cls._convert_timestamp(response.get("cancelling_at")),
            cancelled_at=cls._convert_timestamp(response.get("cancelled_at")),
            total_requests=response.get("request_counts", {}).get("total", 0),
            completed_requests=response.get("request_counts", {}).get("completed", 0),
            failed_requests=response.get("request_counts", {}).get("failed", 0),
            metadata=response.get("metadata"),
        )

    @property
    def is_terminal(self) -> bool:
        """Return True if the batch is in a terminal state."""
        return self.status in (
            BatchStatus.COMPLETED,
            BatchStatus.FAILED,
            BatchStatus.EXPIRED,
            BatchStatus.CANCELLED,
        )

    @property
    def completion_percentage(self) -> float:
        """Return the percentage of completed requests."""
        return (
            (100 * self.completed_requests / self.total_requests)
            if self.total_requests > 0
            else 0.0
        )

    @property
    def has_errors(self) -> bool:
        """Return True if the batch has any errors."""
        return bool(self.error) or self.failed_requests > 0


@dataclass
class BatchListResponse:
    """Response from listing batch jobs."""

    batches: list[BatchInfo]
    first_id: Optional[str] = None
    last_id: Optional[str] = None
    has_more: bool = False


@dataclass
class FileInfo:
    """Information about a file."""

    id: str
    filename: str
    bytes: int
    created_at: int
    purpose: str


@dataclass
class FileListResponse:
    """Response from listing files."""

    files: list[FileInfo]
    has_more: bool = False


_BATCH_PURPOSE = "batch"
_BATCH_ENDPOINT = "/v1/chat/completions"


class AsyncBatchMixin(ABC):
    """Mixin class providing batch processing capabilities.

    Classes using this mixin must implement:
    - _convert_conversation_to_api_input
    - _convert_api_output_to_conversation
    - _get_request_headers

    And must have the following attributes:
    - _remote_params: RemoteParams
    """

    @property
    @abstractmethod
    def _remote_params(self) -> RemoteParams:
        """Remote parameters for API access."""
        pass

    @abstractmethod
    def _convert_conversation_to_api_input(
        self, conversation: Conversation, generation_params: GenerationParams
    ) -> dict[str, Any]:
        """Convert conversation to API input format."""
        pass

    @abstractmethod
    def _convert_api_output_to_conversation(
        self, response: dict[str, Any], original_conversation: Conversation
    ) -> Conversation:
        """Convert API response to conversation."""
        pass

    @abstractmethod
    def _get_request_headers(self, remote_params: RemoteParams) -> dict[str, str]:
        """Get headers for API requests."""
        pass

    async def _upload_batch_file(
        self,
        batch_requests: list[dict],
    ) -> str:
        """Uploads a JSONL file containing batch requests."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".jsonl", delete=False
        ) as tmp:
            with jsonlines.Writer(tmp) as writer:
                for request in batch_requests:
                    writer.write(request)
            tmp_path = tmp.name

        try:
            connector = aiohttp.TCPConnector(limit=self._remote_params.num_workers)
            async with aiohttp.ClientSession(connector=connector) as session:
                headers = self._get_request_headers(self._remote_params)

                form = aiohttp.FormData()
                async with aiofiles.open(tmp_path, "rb") as f:
                    file_data = await f.read()
                    form.add_field("file", file_data, filename="batch_requests.jsonl")
                form.add_field("purpose", _BATCH_PURPOSE)

                async with session.post(
                    f"{self._remote_params.api_url}/files",
                    data=form,
                    headers=headers,
                ) as response:
                    if response.status != 200:
                        raise RuntimeError(
                            f"Failed to upload batch file: {await response.text()}"
                        )
                    data = await response.json()
                    return data["id"]
        finally:
            Path(tmp_path).unlink()

    async def _create_batch(
        self,
        conversations: list[Conversation],
        generation_params: GenerationParams,
    ) -> str:
        """Creates a new batch job."""
        batch_requests = []
        for i, conv in enumerate(conversations):
            api_input = self._convert_conversation_to_api_input(conv, generation_params)
            batch_requests.append(
                {
                    "custom_id": f"request-{i}",
                    "method": "POST",
                    "url": _BATCH_ENDPOINT,
                    "body": api_input,
                }
            )

        file_id = await self._upload_batch_file(batch_requests)

        connector = aiohttp.TCPConnector(limit=self._remote_params.num_workers)
        async with aiohttp.ClientSession(connector=connector) as session:
            headers = self._get_request_headers(self._remote_params)
            async with session.post(
                f"{self._remote_params.api_url}/batches",
                json={
                    "input_file_id": file_id,
                    "endpoint": _BATCH_ENDPOINT,
                    "batch_completion_window": (
                        self._remote_params.batch_completion_window
                    ),
                },
                headers=headers,
            ) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"Failed to create batch: {await response.text()}"
                    )
                data = await response.json()
                return data["id"]

    async def _get_batch_status(
        self,
        batch_id: str,
    ) -> BatchInfo:
        """Gets the status of a batch job."""
        connector = aiohttp.TCPConnector(limit=self._remote_params.num_workers)
        async with aiohttp.ClientSession(connector=connector) as session:
            headers = self._get_request_headers(self._remote_params)
            async with session.get(
                f"{self._remote_params.api_url}/batches/{batch_id}",
                headers=headers,
            ) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"Failed to get batch status: {await response.text()}"
                    )
                data = await response.json()
                return BatchInfo.from_api_response(data)

    async def _list_batches(
        self,
        after: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> BatchListResponse:
        """Lists batch jobs."""
        connector = aiohttp.TCPConnector(limit=self._remote_params.num_workers)
        async with aiohttp.ClientSession(connector=connector) as session:
            headers = self._get_request_headers(self._remote_params)

            params = {}
            if after:
                params["after"] = after
            if limit:
                params["limit"] = str(limit)

            async with session.get(
                f"{self._remote_params.api_url}/batches",
                headers=headers,
                params=params,
            ) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"Failed to list batches: {await response.text()}"
                    )
                data = await response.json()

                batches = [
                    BatchInfo.from_api_response(batch_data)
                    for batch_data in data["data"]
                ]

                return BatchListResponse(
                    batches=batches,
                    first_id=data.get("first_id"),
                    last_id=data.get("last_id"),
                    has_more=data.get("has_more", False),
                )

    async def _get_batch_results_with_mapping(
        self,
        batch_id: str,
        conversations: list[Conversation],
    ) -> list[Conversation]:
        """Gets the results of a completed batch job and maps them to conversations."""
        batch_info = await self._get_batch_status(batch_id)

        if not batch_info.is_terminal:
            raise RuntimeError(
                f"Batch is not in terminal state. Status: {batch_info.status}"
            )

        if batch_info.has_errors:
            if batch_info.error_file_id:
                error_content = await self._download_file(batch_info.error_file_id)
                raise RuntimeError(f"Batch has failed requests: {error_content}")
            raise RuntimeError(f"Batch failed with error: {batch_info.error}")

        if not batch_info.output_file_id:
            raise RuntimeError("No output file available")

        results_content = await self._download_file(batch_info.output_file_id)

        processed_conversations = []
        for line, conv in zip(results_content.splitlines(), conversations):
            result = json.loads(line)
            if result.get("error"):
                raise RuntimeError(f"Batch request failed: {result['error']}")
            processed_conv = self._convert_api_output_to_conversation(
                result["response"]["body"], conv
            )
            processed_conversations.append(processed_conv)
        return processed_conversations

    async def _list_files(
        self,
        purpose: Optional[str] = None,
        limit: Optional[int] = None,
        order: str = "desc",
        after: Optional[str] = None,
    ) -> FileListResponse:
        """Lists files."""
        connector = aiohttp.TCPConnector(limit=self._remote_params.num_workers)
        async with aiohttp.ClientSession(connector=connector) as session:
            headers = self._get_request_headers(self._remote_params)

            params = {"order": order}
            if purpose:
                params["purpose"] = purpose
            if limit:
                params["limit"] = str(limit)
            if after:
                params["after"] = after

            async with session.get(
                f"{self._remote_params.api_url}/files",
                headers=headers,
                params=params,
            ) as response:
                if response.status != 200:
                    raise RuntimeError(f"Failed to list files: {await response.text()}")
                data = await response.json()

                files = [
                    FileInfo(
                        id=file["id"],
                        filename=file["filename"],
                        bytes=file["bytes"],
                        created_at=file["created_at"],
                        purpose=file["purpose"],
                    )
                    for file in data["data"]
                ]

                return FileListResponse(
                    files=files, has_more=len(files) == limit if limit else False
                )

    async def _get_file(
        self,
        file_id: str,
    ) -> FileInfo:
        """Gets information about a file."""
        connector = aiohttp.TCPConnector(limit=self._remote_params.num_workers)
        async with aiohttp.ClientSession(connector=connector) as session:
            headers = self._get_request_headers(self._remote_params)
            async with session.get(
                f"{self._remote_params.api_url}/files/{file_id}",
                headers=headers,
            ) as response:
                if response.status != 200:
                    raise RuntimeError(f"Failed to get file: {await response.text()}")
                data = await response.json()
                return FileInfo(
                    id=data["id"],
                    filename=data["filename"],
                    bytes=data["bytes"],
                    created_at=data["created_at"],
                    purpose=data["purpose"],
                )

    async def _delete_file(
        self,
        file_id: str,
    ) -> bool:
        """Deletes a file."""
        connector = aiohttp.TCPConnector(limit=self._remote_params.num_workers)
        async with aiohttp.ClientSession(connector=connector) as session:
            headers = self._get_request_headers(self._remote_params)
            async with session.delete(
                f"{self._remote_params.api_url}/files/{file_id}",
                headers=headers,
            ) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"Failed to delete file: {await response.text()}"
                    )
                data = await response.json()
                return data.get("deleted", False)

    async def _download_file(
        self,
        file_id: str,
    ) -> str:
        """Downloads a file's content."""
        connector = aiohttp.TCPConnector(limit=self._remote_params.num_workers)
        async with aiohttp.ClientSession(connector=connector) as session:
            headers = self._get_request_headers(self._remote_params)
            async with session.get(
                f"{self._remote_params.api_url}/files/{file_id}/content",
                headers=headers,
            ) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"Failed to download file: {await response.text()}"
                    )
                return await response.text()
