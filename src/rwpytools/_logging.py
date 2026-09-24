"""Logging plumbing that scrubs API keys from rwpytools log records.

A :class:`RedactingFilter` is attached to the package-root logger
(``logging.getLogger("rwpytools")``) on first import. It rewrites any
``api_key=<value>`` token in the formatted message back to
``api_key=***REDACTED***`` so a misconfigured handler (or a user-side
``logging.basicConfig(level=DEBUG)``) can never spill the key into
stderr / files.

The filter is intentionally a no-op when the message contains no
``api_key=`` substring, so the cost on the hot path is a single
substring check.
"""

from __future__ import annotations

import logging
import re

_API_KEY_RE = re.compile(r"(api_key=)[^\s&\"'>]+", re.IGNORECASE)
_REDACTION = r"\1***REDACTED***"


class RedactingFilter(logging.Filter):
    """Strip ``api_key=...`` query-param values from log messages."""

    def filter(self, record: logging.LogRecord) -> bool:
        # ``record.msg`` is the format string; ``record.args`` are the
        # substitutions. Both may carry the secret. We render the final
        # message once, scrub, and overwrite ``msg`` so handlers see
        # the redacted version.
        try:
            rendered = record.getMessage()
        except Exception:
            return True
        if "api_key=" not in rendered.lower():
            return True
        record.msg = _API_KEY_RE.sub(_REDACTION, rendered)
        record.args = ()
        return True


_installed = False


#: Loggers whose records can carry a request URL. rw-api authenticates with
#: an ``api_key`` *query parameter*, and httpx logs every request URL at
#: INFO — so without this, ``logging.basicConfig(level=logging.INFO)``
#: would print the key.
_REDACTED_LOGGERS = ("rwpytools", "httpx", "httpcore")


def install_default_filter() -> None:
    """Attach :class:`RedactingFilter` to the ``rwpytools``, ``httpx`` and
    ``httpcore`` loggers.

    Idempotent — repeated calls do nothing.
    """

    global _installed
    if _installed:
        return
    for name in _REDACTED_LOGGERS:
        logging.getLogger(name).addFilter(RedactingFilter())
    _installed = True


__all__ = ["RedactingFilter", "install_default_filter"]
