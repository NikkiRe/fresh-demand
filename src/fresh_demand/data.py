"""Pinned public data download, integrity checks, and train-only sampling."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import polars as pl

DATASET = "Dingdong-Inc/FreshRetailNet-50K"
REVISION = "c374e85975c82816c495acfa904cc55ab14a47d3"
EXPECTED_MD5 = {
    "train.parquet": "501f47af53495f01592ea8caf1c83978",
    "eval.parquet": "14878135c5b048d4e0dac44ae0282879",
}
KEYS = ["store_id", "product_id"]


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, expected in EXPECTED_MD5.items():
        path = destination / name
        url = f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/data/{name}?download=true"
        if not path.exists():
            temporary = path.with_suffix(".part")
            urllib.request.urlretrieve(url, temporary)
            temporary.replace(path)
        if file_hash(path, "md5") != expected:
            raise ValueError(f"Integrity check failed: {path}; remove and download again")
        records[name] = {"url": url, "bytes": path.stat().st_size, "sha256": file_hash(path), "md5": expected}
    manifest = {"dataset": DATASET, "revision": REVISION, "license": "CC-BY-4.0", "files": records}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def audit(path: Path) -> dict:
    frame = pl.read_parquet(path)
    hourly = frame["hours_sale"].list.to_array(24).to_numpy()
    status = frame["hours_stock_status"].list.to_array(24).to_numpy()
    if hourly.shape != (frame.height, 24) or status.shape != hourly.shape:
        raise ValueError("Expected exactly 24 hourly values per daily record")
    if not np.isfinite(hourly).all() or (hourly < 0).any() or not np.isin(status, [0, 1]).all():
        raise ValueError("Invalid sales or stock status")
    status = status.astype(np.int8)
    duplicates = frame.select(pl.struct(KEYS + ["dt"]).n_unique()).item() != frame.height
    if duplicates:
        raise ValueError("Duplicate store-product-day records")
    stock_count = status[:, 6:22].sum(axis=1)
    if not np.array_equal(stock_count, frame["stock_hour6_22_cnt"].to_numpy()):
        raise ValueError("Operating-hour stock count does not match hours [6:22]")
    return {
        "rows": frame.height,
        "series": frame.select(pl.struct(KEYS).n_unique()).item(),
        "stores": frame["store_id"].n_unique(),
        "products": frame["product_id"].n_unique(),
        "first_date": frame["dt"].min(),
        "last_date": frame["dt"].max(),
        "days": frame["dt"].n_unique(),
        "stockout_hour_fraction": float(status[:, 6:22].mean()),
        "stockout_day_fraction": float((stock_count > 0).mean()),
        "operating_sales_fraction": float(hourly[:, 6:22].sum() / hourly.sum()),
        "daily_hourly_max_discrepancy": float(np.max(np.abs(hourly.sum(axis=1) - frame["sale_amount"].to_numpy()))),
        "positive_sales_in_flagged_operating_hours": int(((hourly[:, 6:22] > 0) & (status[:, 6:22] == 1)).sum()),
        "sha256": file_hash(path),
    }


def select_series(train_path: Path, max_series: int, seed: int) -> pl.DataFrame:
    """Uniform sample of series IDs; no evaluation rows or target values used."""
    pairs = pl.scan_parquet(train_path).select(KEYS).unique().sort(KEYS).collect()
    if max_series > 0 and max_series < pairs.height:
        indices = np.sort(np.random.default_rng(seed).choice(pairs.height, max_series, replace=False))
        pairs = pairs[indices]
    return pairs


def read_subset(path: Path, pairs: pl.DataFrame) -> pl.DataFrame:
    return pl.scan_parquet(path).join(pairs.lazy(), on=KEYS, how="semi").sort(KEYS + ["dt"]).collect()
