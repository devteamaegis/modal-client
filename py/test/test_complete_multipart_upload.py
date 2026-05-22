# Copyright Modal Labs 2022
"""Regression tests for perform_multipart_upload completion retry on S3 503 SlowDown.

Before this fix, _complete_multipart_upload (formerly an inline POST inside
perform_multipart_upload) had no retry logic and no `async with` context manager.
An S3 503 SlowDown on the completion POST would immediately raise ExecutionError,
aborting the entire upload after all parts had already been uploaded successfully.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest

from modal._utils.blob_utils import _complete_multipart_upload
from modal.exception import ExecutionError


def _mock_response(status: int, body: str = "") -> MagicMock:
    resp = MagicMock()
    resp.status = status
    resp.text = AsyncMock(return_value=body)
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)
    return resp


@pytest.mark.asyncio
async def test_complete_multipart_upload_retries_on_503():
    """_complete_multipart_upload retries after a 503 SlowDown response."""
    call_count = 0
    etag = "abc123-2"
    xml_body = f"<ETag>{etag}</ETag>"

    def make_response():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _mock_response(503)
        return _mock_response(200, xml_body)

    session = MagicMock()
    session.post = MagicMock(side_effect=lambda *args, **kwargs: make_response())

    with (
        patch("modal._utils.blob_utils.ClientSessionRegistry.get_session", return_value=session),
        patch("asyncio.sleep", new_callable=AsyncMock),
    ):
        await _complete_multipart_upload("https://s3.example.com/complete", "<xml/>", etag)

    assert call_count == 2, f"Expected 2 POST attempts (1 × 503 + 1 × 200), got {call_count}"


@pytest.mark.asyncio
async def test_complete_multipart_upload_succeeds_on_first_try():
    """_complete_multipart_upload completes without retry when S3 returns 200 immediately."""
    etag = "deadbeef-3"
    xml_body = f"<ETag>{etag}</ETag>"

    session = MagicMock()
    session.post = MagicMock(return_value=_mock_response(200, xml_body))

    with patch("modal._utils.blob_utils.ClientSessionRegistry.get_session", return_value=session):
        await _complete_multipart_upload("https://s3.example.com/complete", "<xml/>", etag)

    assert session.post.call_count == 1


@pytest.mark.asyncio
async def test_complete_multipart_upload_raises_on_non_retryable_error():
    """_complete_multipart_upload raises ExecutionError immediately on non-503 errors (e.g. 403)."""
    session = MagicMock()
    session.post = MagicMock(return_value=_mock_response(403, "Access Denied"))

    with patch("modal._utils.blob_utils.ClientSessionRegistry.get_session", return_value=session):
        with pytest.raises(ExecutionError, match="403"):
            await _complete_multipart_upload("https://s3.example.com/complete", "<xml/>", "etag-1")


@pytest.mark.asyncio
async def test_complete_multipart_upload_sleeps_on_503():
    """_complete_multipart_upload calls asyncio.sleep(1) on 503 before retrying."""
    etag = "abc123-1"
    xml_body = f"<ETag>{etag}</ETag>"
    call_count = 0

    def make_response():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _mock_response(503)
        return _mock_response(200, xml_body)

    session = MagicMock()
    session.post = MagicMock(side_effect=lambda *args, **kwargs: make_response())

    sleep_calls = []

    async def track_sleep(delay):
        sleep_calls.append(delay)

    with (
        patch("modal._utils.blob_utils.ClientSessionRegistry.get_session", return_value=session),
        patch("asyncio.sleep", side_effect=track_sleep),
    ):
        await _complete_multipart_upload("https://s3.example.com/complete", "<xml/>", etag)

    assert 1 in sleep_calls, f"Expected asyncio.sleep(1) call after 503; got sleep_calls={sleep_calls}"


@pytest.mark.asyncio
async def test_complete_multipart_upload_hash_mismatch_raises():
    """_complete_multipart_upload raises ExecutionError when the ETag is absent from the S3 response."""
    session = MagicMock()
    session.post = MagicMock(return_value=_mock_response(200, "<ETag>wrong-etag-9</ETag>"))

    with patch("modal._utils.blob_utils.ClientSessionRegistry.get_session", return_value=session):
        with pytest.raises(ExecutionError, match="Hash mismatch"):
            await _complete_multipart_upload(
                "https://s3.example.com/complete", "<xml/>", "expected-etag-5"
            )
