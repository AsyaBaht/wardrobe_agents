"""One structured entry point to Claude.

Every agent in this package calls Claude through :class:`StructuredLLM`, which
takes a Pydantic model and returns a validated instance of it. Free text is
never parsed, pattern-matched, or regex'd anywhere in this codebase - if Claude's
answer does not satisfy the schema, the call fails loudly instead of degrading
into a guess.

The mechanism is the Messages API's structured outputs (``messages.parse`` with
``output_format=<PydanticModel>``): the model is constrained to emit JSON
matching the schema, and the SDK validates it before returning. This is the
current API for "the response must match this schema exactly"; it supersedes the
older idiom of declaring a single tool and forcing ``tool_choice`` onto it, and
unlike that idiom it composes with adaptive thinking on every platform.

The API key is read from the environment by the SDK at call time (see
``Settings.api_key_env_var``); it is never stored on an object, written to a
report, or included in an error message.

Author: Anastasiia Bakhtoiarova
"""

from __future__ import annotations

import base64
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, Sequence, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from config.settings import Settings, settings as default_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

MEDIA_TYPES: dict[str, str] = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

ContentBlocks = str | list[dict[str, Any]]


class LLMError(RuntimeError):
    """Any failure to obtain a valid structured response from Claude."""


class MissingAPIKeyError(LLMError):
    """No Claude API key is available in the environment."""


def image_block(path: Path | str) -> dict[str, Any]:
    """Build a base64 image content block from a local file."""
    path = Path(path)
    if not path.exists():
        raise LLMError(f"No such image: {path}")

    media_type = MEDIA_TYPES.get(path.suffix.lower())
    if media_type is None:
        supported = ", ".join(sorted(MEDIA_TYPES))
        raise LLMError(f"Unsupported image type {path.suffix!r}. Supported: {supported}")

    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}


def text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


@dataclass(slots=True)
class LLMRequest(Generic[T]):
    """One structured call, for use with :meth:`StructuredLLM.call_batch`."""

    system: str
    content: ContentBlocks
    response_model: type[T]
    purpose: str
    model: str | None = None
    max_tokens: int | None = None
    effort: str | None = None


@dataclass(slots=True)
class BatchResult(Generic[T]):
    """Per-request outcome of a batch. One failure does not sink the batch - a
    single unreadable photo should not abandon the other nine."""

    request: LLMRequest[T]
    value: T | None = None
    error: Exception | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def unwrap(self) -> T:
        if self.error is not None:
            raise self.error
        assert self.value is not None
        return self.value


class StructuredLLM:
    """Thin, testable wrapper around the Anthropic client.

    Agents accept one of these by constructor injection, so tests substitute a
    fake with the same ``call`` / ``call_batch`` surface and never touch the
    network.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        client: Any | None = None,
    ) -> None:
        self.settings = settings or default_settings
        self._client = client

    # ---- availability ---------------------------------------------------

    def is_configured(self) -> bool:
        """True if a call could be made right now (key present, or a client injected)."""
        return self._client is not None or self.settings.api_key() is not None

    def require_configured(self, what: str) -> None:
        if not self.is_configured():
            raise MissingAPIKeyError(
                f"{what} needs the Claude API. Set {self.settings.api_key_env_var} in your "
                "environment (or run `ant auth login`) and try again."
            )

    @property
    def client(self) -> Any:
        if self._client is None:
            if self.settings.api_key() is None:
                raise MissingAPIKeyError(
                    f"No Claude API key found. Set {self.settings.api_key_env_var} in your environment."
                )
            self._client = anthropic.Anthropic(timeout=self.settings.llm_timeout_seconds)
        return self._client

    # ---- calls ----------------------------------------------------------

    def call(
        self,
        *,
        system: str,
        content: ContentBlocks,
        response_model: type[T],
        purpose: str,
        model: str | None = None,
        max_tokens: int | None = None,
        effort: str | None = None,
    ) -> T:
        """Make one structured call and return a validated ``response_model``.

        Raises :class:`LLMError` (never returns a partial or unparsed result).
        """
        self.require_configured(purpose)
        model = model or self.settings.claude_model
        message_content: Any = content if isinstance(content, str) else list(content)

        logger.debug("Claude call [%s] model=%s schema=%s", purpose, model, response_model.__name__)

        try:
            response = self.client.messages.parse(
                model=model,
                max_tokens=max_tokens or self.settings.llm_max_tokens,
                system=system,
                messages=[{"role": "user", "content": message_content}],
                thinking={"type": "adaptive"},
                output_config={"effort": effort or self.settings.llm_effort},
                output_format=response_model,
            )
        except anthropic.AuthenticationError as exc:
            raise MissingAPIKeyError(
                f"Claude rejected the credentials in {self.settings.api_key_env_var} while {purpose}."
            ) from exc
        except anthropic.NotFoundError as exc:
            raise LLMError(f"Model {model!r} not found or not available to this account.") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError(f"Rate limited by the Claude API while {purpose}. Retry shortly.") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"Claude API error {exc.status_code} while {purpose}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError(f"Could not reach the Claude API while {purpose}: {exc}") from exc
        except ValidationError as exc:
            # The SDK validates client-side, so an out-of-range value or a response
            # truncated at max_tokens surfaces here rather than as an API error.
            first = exc.errors()[0]
            where = ".".join(str(part) for part in first["loc"]) or "response"
            raise LLMError(
                f"Claude's response did not match {response_model.__name__} while {purpose} "
                f"({exc.error_count()} problem(s); first at {where}: {first['msg']})."
            ) from exc

        # Safety classifiers can decline with HTTP 200; check before reading content.
        if response.stop_reason == "refusal":
            detail = getattr(response, "stop_details", None)
            category = getattr(detail, "category", None) or "unspecified"
            raise LLMError(f"Claude declined the request while {purpose} (category: {category}).")

        parsed = response.parsed_output
        if parsed is None:
            raise LLMError(
                f"Claude returned no structured output while {purpose} "
                f"(stop_reason={response.stop_reason}). Expected {response_model.__name__}."
            )
        return parsed

    def call_batch(self, requests: Sequence[LLMRequest[T]]) -> list[BatchResult[T]]:
        """Run several structured calls concurrently, bounded by
        ``settings.max_concurrent_llm_calls``. Results keep input order."""
        if not requests:
            return []

        workers = max(1, min(self.settings.max_concurrent_llm_calls, len(requests)))

        def run(req: LLMRequest[T]) -> BatchResult[T]:
            try:
                value = self.call(
                    system=req.system,
                    content=req.content,
                    response_model=req.response_model,
                    purpose=req.purpose,
                    model=req.model,
                    max_tokens=req.max_tokens,
                    effort=req.effort,
                )
                return BatchResult(request=req, value=value)
            except Exception as exc:  # surfaced per-item, not raised through the batch
                logger.warning("Batch call failed [%s]: %s", req.purpose, exc)
                return BatchResult(request=req, error=exc)

        if workers == 1:
            return [run(req) for req in requests]

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="wardrobe-llm") as pool:
            return list(pool.map(run, requests))


@dataclass(slots=True)
class RecordedCall:
    """A call captured by :class:`FakeLLM`, for assertions in tests."""

    system: str
    content: ContentBlocks
    response_model: type[BaseModel]
    purpose: str


class FakeLLM(StructuredLLM):
    """Test double that returns queued responses instead of calling the API.

    Lives beside the real implementation on purpose: it is the contract every
    agent is tested against, so it should drift only when ``call`` drifts.
    """

    def __init__(self, responses: Sequence[BaseModel | Exception] | None = None) -> None:
        super().__init__()
        self.responses: list[BaseModel | Exception] = list(responses or [])
        self.calls: list[RecordedCall] = field(default_factory=list)  # type: ignore[assignment]
        self.calls = []

    def is_configured(self) -> bool:
        return True

    def call(  # type: ignore[override]
        self,
        *,
        system: str,
        content: ContentBlocks,
        response_model: type[T],
        purpose: str,
        model: str | None = None,
        max_tokens: int | None = None,
        effort: str | None = None,
    ) -> T:
        self.calls.append(
            RecordedCall(system=system, content=content, response_model=response_model, purpose=purpose)
        )
        if not self.responses:
            raise LLMError(f"FakeLLM has no queued response for {purpose!r}.")
        nxt = self.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        if not isinstance(nxt, response_model):
            raise LLMError(
                f"FakeLLM queued a {type(nxt).__name__} but {purpose} expects {response_model.__name__}."
            )
        return nxt
