import json
from typing import Any, Dict

from aidial_sdk.exceptions import HTTPException as DialException
from aidial_sdk.exceptions import InternalServerError
from fastapi import HTTPException as FastAPIException
from fastapi import Request
from fastapi.responses import Response
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    OpenAIError,
)
from typing_extensions import override


class PlainDialException(DialException):
    content: Any

    def __init__(
        self,
        *,
        content: Any,
        status_code: int,
        headers: Dict[str, str] | None,
    ) -> None:
        super().__init__(
            message=str(content),
            status_code=status_code,
            headers=headers,
        )
        self.content = content

    @override
    def to_fastapi_response(self) -> Response:  # type: ignore
        return Response(
            status_code=self.status_code,
            content=self.content,
            headers=self.headers,
        )

    @override
    def to_fastapi_exception(self) -> FastAPIException:
        return FastAPIException(
            status_code=self.status_code,
            detail=self.content,
            headers=self.headers,
        )


def _parse_dial_exception(
    *, status_code: int, headers: Dict[str, str], content: Any
) -> DialException | None:
    if isinstance(content, str):
        try:
            obj = json.loads(content)
        except Exception:
            return None
    else:
        obj = content

    if (
        isinstance(obj, dict)
        and (error := obj.get("error"))
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

    return None


def to_dial_exception(exc: Exception) -> DialException:
    if isinstance(exc, APIStatusError):
        # Non-streaming errors reported by `openai` library via this exception
        r = exc.response
        httpx_headers = r.headers

        # httpx library (used by openai) automatically sets
        # "Accept-Encoding:gzip,deflate" header in requests to the upstream.
        # Therefore, we may receive from the upstream gzip-encoded
        # response along with "Content-Encoding:gzip" header.
        # We either need to encode the response, or
        # remove the "Content-Encoding" header.
        if "Content-Encoding" in httpx_headers:
            del httpx_headers["Content-Encoding"]

        headers = {k.decode(): v.decode() for k, v in httpx_headers.raw}
        status_code = r.status_code
        content = r.text

        return _parse_dial_exception(
            status_code=status_code,
            headers=headers,
            content=content,
        ) or PlainDialException(
            status_code=status_code,
            headers=headers,
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

        headers = {}
        content = {"error": exc.body or {}}

        return _parse_dial_exception(
            status_code=status_code,
            headers=headers,
            content=content,
        ) or PlainDialException(
            status_code=status_code,
            headers=headers,
            content=content,
        )

    if isinstance(exc, DialException):
        return exc

    return InternalServerError(str(exc))


def openai_exception_handler(request: Request, exc: Exception) -> Response:
    assert isinstance(exc, OpenAIError)
    return to_dial_exception(exc).to_fastapi_response()
