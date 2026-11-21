"""Loading Spider and making the train / validation / test splits.

* Questions + gold SQL come from the Hugging Face copy `xlangai/spider`
  (train = 7,000, validation = 1,034 = the official *dev* set, which we use as TEST).
  The HF dev set is the newer, corrected version of Spider dev (44 queries fixed vs. the 2020 zip).
* The SQLite database files come from the original Spider release (Kaggle dataset
  `jeromeblanchet/yale-universitys-spider-10-nlp-dataset`, folder `spider/database/`).
* Our *validation* split is carved out of train and is database-disjoint: whole databases
  are held out, like the dev set (whose databases never appear in train).
"""
from __future__ import annotations

import os
import random
from pathlib import Path

HF_DATASET = "xlangai/spider"


def load_spider(split: str) -> list[dict]:
    """Return a list of {db_id, question, query} dicts. split: 'train' or 'test' (= Spider dev)."""
    from datasets import load_dataset  # imported lazily: not needed by the demo

    hf_split = {"train": "train", "test": "validation"}[split]
    ds = load_dataset(HF_DATASET, split=hf_split)
    return [
        {"id": f"{split}_{i}", "db_id": e["db_id"], "question": e["question"], "query": e["query"]}
        for i, e in enumerate(ds)
    ]


def find_spider_root(start: str | None = None) -> Path:
    """Locate the Spider folder that contains `database/` and `tables.json`.

    Order: $SPIDER_DIR, then a recursive search under `start` (default /kaggle/input).
    """
    env = os.environ.get("SPIDER_DIR")
    if env and (Path(env) / "database").is_dir():
        return Path(env)
    base = Path(start or "/kaggle/input")
    for tables in base.rglob("tables.json"):
        if (tables.parent / "database").is_dir():
            return tables.parent
    raise FileNotFoundError(f"No Spider folder with database/ + tables.json found under {base}")


def db_path(spider_root: Path | str, db_id: str) -> Path:
    return Path(spider_root) / "database" / db_id / f"{db_id}.sqlite"


def split_train_val(train: list[dict], val_frac: float = 0.05, seed: int = 42):
    """Database-disjoint split: hold out whole databases until >= val_frac of the examples.

    Deterministic for a given seed. Returns (train_examples, val_examples, val_db_ids).
    """
    db_ids = sorted({e["db_id"] for e in train})
    random.Random(seed).shuffle(db_ids)
    target = val_frac * len(train)
    counts: dict[str, int] = {}
    for e in train:
        counts[e["db_id"]] = counts.get(e["db_id"], 0) + 1

    val_dbs, n = set(), 0
    for db in db_ids:
        if n >= target:
            break
        val_dbs.add(db)
        n += counts[db]
    tr = [e for e in train if e["db_id"] not in val_dbs]
    va = [e for e in train if e["db_id"] in val_dbs]
    return tr, va, sorted(val_dbs)
