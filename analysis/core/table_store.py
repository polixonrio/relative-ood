"""Small table and analysis-artifact IO helpers."""

from __future__ import annotations

import json

import pandas as pd


def _json_default(value):
    """json.dump default: map pd.NA to None and array-likes to lists."""
    if value is pd.NA:
        return None
    if hasattr(value, 'tolist'):
        return value.tolist()
    raise TypeError(f'{type(value).__name__} is not JSON serializable')


def write_dataframe(frame: pd.DataFrame, path) -> None:
    """Write a DataFrame as parquet and create its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def write_json(payload: dict, path) -> None:
    """Write a JSON payload and create its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2, default=_json_default)


def write_artifacts(output_dir, tables: dict[str, pd.DataFrame] | None = None, summary: dict | None = None, csv: bool | set[str] = False, summary_name: str = 'summary.json'):
    """Write named parquet tables, optional CSV mirrors, and optional summary JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in (tables or {}).items():
        write_dataframe(frame, output_dir / f'{name}.parquet')
        if csv is True or (isinstance(csv, set) and name in csv):
            frame.to_csv(output_dir / f'{name}.csv', index=False)
    if summary is not None:
        write_json(summary, output_dir / summary_name)


def figures_dir(output_dir):
    """Create and return the standard figures directory for an analysis output."""
    path = output_dir / 'figures'
    path.mkdir(parents=True, exist_ok=True)
    return path


def iter_dataframes(path, columns: list[str] | None = None):
    """Yield one or more parquet frames from a table path or parts directory."""
    if path.is_dir():
        for part_path in sorted(path.glob('*.parquet')):
            yield pd.read_parquet(part_path, columns=columns)
        return
    yield pd.read_parquet(path, columns=columns)
