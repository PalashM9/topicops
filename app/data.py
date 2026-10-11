import argparse
import hashlib
import json

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

from app.common import CATEGORIES, DATA, SEED, ensure_dirs, sha256, write_json


def validate(rows):
    if not rows:
        raise ValueError("Dataset is empty")
    for row in rows:
        if not isinstance(row.get("text"), str) or not row["text"].strip():
            raise ValueError("Invalid text")
        if row.get("label") not in CATEGORIES:
            raise ValueError("Invalid label")
        if row.get("id") != hashlib.sha256(row["text"].encode()).hexdigest():
            raise ValueError("Invalid text fingerprint")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate documents")
    if {row["label"] for row in rows} != set(CATEGORIES):
        raise ValueError("Missing class")


def save_rows(name, rows):
    validate(rows)
    path = DATA / f"{name}.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return {"rows": len(rows), "sha256": sha256(path)}


def load_rows(name):
    rows = [json.loads(line) for line in (DATA / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()]
    validate(rows)
    return rows


def prepare():
    ensure_dirs()
    seen = set()
    splits = {}
    dropped = {}
    for subset in ["train", "test"]:
        raw = fetch_20newsgroups(
            subset=subset, categories=CATEGORIES,
            remove=("headers", "footers", "quotes"), random_state=SEED,
            data_home=str(DATA / "cache"),
        )
        rows = []
        for text, target in zip(raw.data, raw.target):
            text = text.strip()
            fingerprint = hashlib.sha256(text.encode()).hexdigest()
            if not text or fingerprint in seen:
                continue
            seen.add(fingerprint)
            rows.append({"id": fingerprint, "text": text, "label": raw.target_names[int(target)]})
        splits[subset] = rows
        dropped[subset] = len(raw.data) - len(rows)
    indices = np.arange(len(splits["train"]))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, random_state=SEED,
        stratify=[row["label"] for row in splits["train"]],
    )
    output = {
        "train": [splits["train"][i] for i in train_idx],
        "validation": [splits["train"][i] for i in val_idx],
        "test": splits["test"],
    }
    manifest = {name: save_rows(name, rows) for name, rows in output.items()}
    write_json(DATA / "manifest.json", {"splits": manifest, "seed": SEED, "dropped": dropped})
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    argparse.ArgumentParser(description="Download, clean, split and fingerprint data").parse_args()
    prepare()
