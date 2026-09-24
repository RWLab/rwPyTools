"""Per-pod data accessors.

Each pod corresponds to one Robot Wealth research pod and is exported
in two flavours:

* ``Async{Name}Pod`` — async methods, used by
  :class:`rwpytools.AsyncClient`.
* ``{Name}Pod``       — synchronous wrapper used by
  :class:`rwpytools.Client`. Each method delegates to its async
  counterpart via :func:`rwpytools.http.run_sync`.

Both surfaces expose the same method names and signatures
(plus/minus the ``async``). Method names mirror the equivalent
rwRTools function with the ``<pod>_get_`` prefix dropped.
"""

from ._base import AsyncPodAccessor, SyncPodAccessor
from .crypto import AsyncCryptoPod, CryptoPod
from .equity import AsyncEquityFactorsPod, EquityFactorsPod
from .fx import AsyncFxPod, FxPod
from .macro import AsyncMacroPod, MacroPod

__all__ = [
    "AsyncCryptoPod",
    "AsyncEquityFactorsPod",
    "AsyncFxPod",
    "AsyncMacroPod",
    "AsyncPodAccessor",
    "CryptoPod",
    "EquityFactorsPod",
    "FxPod",
    "MacroPod",
    "SyncPodAccessor",
]
