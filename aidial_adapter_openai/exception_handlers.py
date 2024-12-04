from typing import Dict

from aidial_sdk.exceptions import HTTPException as DialException
from aidial_sdk.exceptions import InternalServerError
from fastapi import Request
from fastapi.responses import Response
from openai import (APIConnectionError, APIError, APIStatusError,
                    APITimeoutError, OpenAIError)

from aidial_adapter_openai.utils.adapter_exception import (
    AdapterException, ResponseWrapper, parse_adapter_exception)
from aidial_adapter_openai.utils.log_config import logger


def _parse_dial_exception(
    status_code: int,
    content: dict | str,
    headers: Dict[str, str] | None = None,
) -> DialException:
    if (
        isinstance(content, dict)
        and (error := content.get("error"))
        and isinstance(error, dict)
    ):
        message = error.get("message") or "Unknown error"
        code = error.get("code")
        type = error.get("type")
        param = error.get("param")
        display_message = error.get("display_message")

        return DialException(
            status_code=status_code,
            message=message,
            type=type,
            param=param,
            code=code,
            display_message=display_message,
            headers=headers,
        )
    else:
        return DialException(
            status_code=status_code,
            message=str(content),
            headers=headers,
        )


def to_dial_exception(exc: Exception) -> DialException:
    if isinstance(exc, APIStatusError):
        # Non-streaming errors reported by `openai` library via this exception

        r = exc.response
        headers = r.headers

        # httpx library (used by openai) automatically sets
        # "Accept-Encoding:gzip,deflate" header in requests to the upstream.
        # Therefore, we may receive from the upstream gzip-encoded
        # response along with "Content-Encoding:gzip" header.
        # We either need to encode the response, or
        # remove the "Content-Encoding" header.
        if "Content-Encoding" in headers:
            del headers["Content-Encoding"]

        plain_headers = {k.decode(): v.decode() for k, v in headers.raw}

        try:
            content = r.json()
        except Exception:
            content = r.text

        return _parse_dial_exception(
            status_code=r.status_code,
            headers=plain_headers,
            content=content,
        )

    if isinstance(exc, APITimeoutError):
        return DialException(
            status_code=504,
            type="timeout",
            message="Request timed out",
            display_message="Request timed out. Please try again later.",
        )

    if isinstance(exc, APIConnectionError):
        return DialException(
            status_code=502,
            type="connection",
            message="Error communicating with OpenAI",
            display_message="OpenAI server is not responsive. Please try again later.",
        )

    if isinstance(exc, APIError):
        # Streaming errors reported by `openai` library via this exception
        status_code: int = 500
        if exc.code:
            try:
                status_code = int(exc.code)
            except Exception:
                pass

        return _parse_dial_exception(
            status_code=status_code,
            headers={},
            content={"error": exc.body or {}},
        )

    if isinstance(exc, DialException):
        return exc

    return InternalServerError(str(exc))


def openai_exception_handler(request: Request, exc: Exception) -> Response:
    assert isinstance(exc, OpenAIError)
    return to_dial_exception(exc).to_fastapi_response()
