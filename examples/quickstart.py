"""Quickstart: one call for history plus real time.

``client.get()`` serves history from the dataset's bulk export (downloaded
once, then cached on disk) and tops it up from the live API with every row
newer than the export. ``result.describe_route()`` says exactly which rows
came from where.

Run::

    RWPYTOOLS_API_KEY=... python examples/quickstart.py
"""

from __future__ import annotations

import rwpytools


def main() -> None:
    # The key comes from RWPYTOOLS_API_KEY, or a masked prompt in a terminal.
    with rwpytools.Client() as client:
        print(client.status())

        result = client.get("vix", start="2024-01-01")

        print(result.describe_route())
        print(f"{len(result)} rows, columns: {result.columns}")
        print(f"export watermark: {result.watermark}")

        df = result.df
        print(df.tail())


if __name__ == "__main__":
    main()
