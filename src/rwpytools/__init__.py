"""Robot Wealth research data client.

Python equivalent of `rwRTools <https://github.com/RWLab/rwRtools>`_.
History comes from rw-api's bulk exports (signed Cloud CDN URLs, cached on
disk); anything newer than an export's watermark comes from the matching
live JSON endpoint. :meth:`Client.get` routes between them automatically.

Top-level surface:

* :class:`Client`         — sync, primary entry point.
* :class:`AsyncClient`    — async-native sibling with full pod parity.
* :class:`Result`         — rows plus the routing decision that produced them.
* :class:`RoutePlan`      — that routing decision (see :meth:`Client.explain`).
* :class:`Source`         — ``"auto"`` / ``"bulk"`` / ``"live"`` override.
* :data:`DATASETS`        — every routable dataset and its bulk / live sources.
* :class:`ClientConfig`   — explicit configuration when env vars aren't enough.
* :class:`FetchRequest`   — value type for batched bulk fetches.
* :class:`FetchResult`    — value type returned by :class:`bulk.BulkClient`.
* :mod:`rwpytools.errors` — the exception hierarchy.
"""

from __future__ import annotations

from . import errors
from ._logging import install_default_filter as _install_default_filter
from ._version import __version__
from .bulk import BulkClient, FetchRequest, FetchResult
from .client import AsyncClient, Client
from .config import ClientConfig
from .datasets import DATASETS, DatasetSpec
from .live import AsyncLiveClient, LiveClient
from .pods import (
    AsyncCryptoPod,
    AsyncEquityFactorsPod,
    AsyncFxPod,
    AsyncMacroPod,
    CryptoPod,
    EquityFactorsPod,
    FxPod,
    MacroPod,
)
from .result import Result
from .routing import RoutePlan, Source

# Make sure no rwpytools log record can leak an api_key, no matter what
# logging config the host process has set up. Runs at import time so
# any later ``logging.basicConfig(level=DEBUG)`` already inherits the
# filter on the package-root logger.
_install_default_filter()

__all__ = [
    "DATASETS",
    "AsyncClient",
    "AsyncCryptoPod",
    "AsyncEquityFactorsPod",
    "AsyncFxPod",
    "AsyncLiveClient",
    "AsyncMacroPod",
    # Subsystems advanced users may import
    "BulkClient",
    # Primary entry points
    "Client",
    "ClientConfig",
    # Pod accessors (mainly useful as type hints)
    "CryptoPod",
    "DatasetSpec",
    "EquityFactorsPod",
    # Value types
    "FetchRequest",
    "FetchResult",
    "FxPod",
    "LiveClient",
    "MacroPod",
    "Result",
    "RoutePlan",
    "Source",
    "__version__",
    # Misc
    "errors",
]
