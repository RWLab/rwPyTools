"""Wire-format Pydantic models for rw-api responses.

Mirrors the OpenAPI definitions in ``rw-api/api-paths/app/openapi/openapi.yaml``.
Keeping them in one place gives a single audit point if the server
contract changes.

Field naming
------------

The server JSON uses ``schema`` for the OHLC variant name (``ohlcv-1h``,
``calendar``, ``snapshot``, ...). Python can't expose an attribute
called ``schema`` on a :class:`pydantic.BaseModel` subclass because
that name is reserved on the base class. We declare the field as
``schema_name`` and use an alias so deserialization accepts the wire
``schema`` key transparently; callers who actually need the value
read ``model.schema_name``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class _Frozen(BaseModel):
    """Base for response models — immutable, ignores extras for forward-compat.

    ``populate_by_name=True`` lets tests construct instances using the
    Python field name (``schema_name=...``) as well as the wire alias
    (``schema=...``).
    """

    model_config = ConfigDict(frozen=True, extra="ignore", populate_by_name=True)


# ---- /v1/{ns}/datasets ---------------------------------------------------


class DatasetSchemaEntry(_Frozen):
    schema_name: str = Field(alias="schema")
    format: str


class DatasetCatalogEntry(_Frozen):
    dataset: str
    requires_symbol: bool = False
    schemas: list[DatasetSchemaEntry]


class DatasetCatalogResponse(_Frozen):
    success: bool
    namespace: str
    datasets: list[DatasetCatalogEntry]


# ---- /v1/{ns}/file -------------------------------------------------------


class PresignedUrl(_Frozen):
    namespace: str
    dataset: str
    schema_name: str = Field(alias="schema")
    symbol: str | None = None
    format: str
    bucket: str
    object: str
    size: int
    expires_in: int
    url: str


class PresignedUrlResponse(_Frozen):
    success: bool
    data: PresignedUrl


# ---- /v1/fx/symbols ------------------------------------------------------


class SymbolsResponse(_Frozen):
    success: bool
    namespace: str
    dataset: str
    schema_name: str = Field(alias="schema")
    symbols: list[str]
    count: int


__all__ = [
    "DatasetCatalogEntry",
    "DatasetCatalogResponse",
    "DatasetSchemaEntry",
    "PresignedUrl",
    "PresignedUrlResponse",
    "SymbolsResponse",
]
