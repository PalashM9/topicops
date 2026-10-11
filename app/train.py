import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import mlflow
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from app.common import DATA, MODELS, SEED, STATE, ensure_dirs, read_json, sha256, write_json
from app.data import load_rows


def new_pipeline(c=1.0):
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=30000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("classifier", LogisticRegression(C=c, max_iter=1000, random_state=SEED)),
    ])


def evaluate(model, rows):
    targets = [row["label"] for row in rows]
    predicted = model.predict([row["text"] for row in rows])
    return {
        "accuracy": float(accuracy_score(targets, predicted)),
        "macro_f1": float(f1_score(targets, predicted, average="macro", zero_division=0)),
        "classification_report": classification_report(targets, predicted, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(targets, predicted, labels=model.classes_).tolist(),
        "class_order": model.classes_.tolist(),
    }


def signals(model, texts):
    vectors = model.named_steps["tfidf"].transform(texts)
    probability = model.predict_proba(texts)
    return {
        "length": np.asarray([len(text) for text in texts], dtype=float),
        "confidence": probability.max(axis=1),
        "empty_vector": (vectors.getnnz(axis=1) == 0).astype(float),
    }


def reference_profile(model, texts):
    values = signals(model, texts)
    edges = {"length": [0, 100, 300, 1000, 3000, 10000, 1000000000],
             "confidence": [0, .3, .5, .7, .9, 1.000001]}
    return {
        "histograms": {name: {"edges": bins, "counts": np.histogram(values[name], bins=bins)[0].tolist()}
                       for name, bins in edges.items()},
        "empty_vector_rate": float(values["empty_vector"].mean()),
        "rows": len(texts),
    }


def train():
    ensure_dirs()
    manifest = read_json(DATA / "manifest.json")
    for name, info in manifest["splits"].items():
        if sha256(DATA / f"{name}.jsonl") != info["sha256"]:
            raise ValueError(f"Data changed: {name}; run data preparation again")
    train_rows = load_rows("train")
    validation = load_rows("validation")
    mlflow.set_tracking_uri(f"sqlite:///{STATE / 'mlflow.db'}")
    mlflow.set_experiment("topic-router")
    candidates = []
    for c in [0.5, 1.0, 2.0]:
        with mlflow.start_run(run_name=f"logreg-C-{c}") as run:
            mlflow.log_params({"C": c, "seed": SEED, "features": 30000, "train_rows": len(train_rows)})
            mlflow.set_tag("data_train_sha256", manifest["splits"]["train"]["sha256"])
            model = new_pipeline(c)
            model.fit([row["text"] for row in train_rows], [row["label"] for row in train_rows])
            metrics = evaluate(model, validation)
            mlflow.log_metrics({f"validation_{key}": metrics[key] for key in ["accuracy", "macro_f1"]})
            folder = MODELS / run.info.run_id
            folder.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, folder / "model.joblib")
            write_json(folder / "validation.json", metrics)
            write_json(folder / "reference.json", reference_profile(model, [row["text"] for row in validation]))
            write_json(folder / "metadata.json", {
                "version": run.info.run_id, "created_at": datetime.now(timezone.utc).isoformat(),
                "C": c, "dataset": manifest, "model_sha256": sha256(folder / "model.joblib"),
                "classes": model.classes_.tolist(),
            })
            for name in ["model.joblib", "validation.json", "reference.json", "metadata.json"]:
                mlflow.log_artifact(str(folder / name), artifact_path="candidate")
            candidates.append((metrics["macro_f1"], run.info.run_id))
    score, version = max(candidates)
    write_json(MODELS / "candidate.json", {"version": version, "validation_macro_f1": score})
    print(json.dumps({"candidate": version, "validation_macro_f1": score}, indent=2))


if __name__ == "__main__":
    argparse.ArgumentParser(description="Train three candidates; select by validation macro F1").parse_args()
    train()
