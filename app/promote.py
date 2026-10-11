import argparse
import json

import joblib

from app.common import MODELS, read_json, sha256, write_json
from app.data import load_rows
from app.train import evaluate


def promote(min_f1=0.70, max_regression=0.01):
    candidate = read_json(MODELS / "candidate.json")
    folder = MODELS / candidate["version"]
    metadata = read_json(folder / "metadata.json")
    if sha256(folder / "model.joblib") != metadata["model_sha256"]:
        raise ValueError("Candidate artifact hash mismatch")
    metric = read_json(folder / "validation.json")["macro_f1"]
    reason = None
    if metric < min_f1:
        reason = f"Validation F1 {metric:.3f} is below {min_f1}"
    production = MODELS / "production.json"
    if production.exists():
        old = read_json(production)
        if metric < old["validation_macro_f1"] - max_regression:
            reason = "Candidate regresses beyond allowed tolerance"
    if reason:
        raise SystemExit(f"Promotion rejected: {reason}")
    # Test metrics are reported after selection; they do not drive the gate.
    test_file = folder / "test.json"
    if not test_file.exists():
        model = joblib.load(folder / "model.joblib")
        write_json(test_file, evaluate(model, load_rows("test")))
    if production.exists():
        write_json(MODELS / "previous.json", read_json(production))
    write_json(production, {
        "version": candidate["version"], "validation_macro_f1": metric,
        "model_sha256": metadata["model_sha256"],
    })
    print(json.dumps({"production": read_json(production), "test": read_json(test_file)}, indent=2))


def rollback():
    previous = MODELS / "previous.json"
    if not previous.exists():
        raise SystemExit("No previous production model")
    target = read_json(previous)
    current = read_json(MODELS / "production.json")
    write_json(MODELS / "production.json", target)
    write_json(previous, current)
    print("Rolled back; restart the API to load the model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-f1", type=float, default=0.70)
    parser.add_argument("--max-regression", type=float, default=0.01)
    parser.add_argument("--rollback", action="store_true")
    args = parser.parse_args()
    if args.rollback:
        rollback()
    else:
        promote(args.min_f1, args.max_regression)
