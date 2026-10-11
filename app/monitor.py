import argparse
import json

import numpy as np

from app.common import MODELS, STATE, read_json, write_json
from app.store import recent


def psi(reference, current):
    reference = np.asarray(reference, dtype=float) + 0.5
    current = np.asarray(current, dtype=float) + 0.5
    reference /= reference.sum()
    current /= current.sum()
    return float(np.sum((current - reference) * np.log(current / reference)))


def monitor(limit=1000, minimum=100, threshold=0.2):
    pointer = read_json(MODELS / "production.json")
    reference = read_json(MODELS / pointer["version"] / "reference.json")
    rows = recent(pointer["version"], limit)
    report = {"version": pointer["version"], "rows": len(rows), "alerts": []}
    if len(rows) < minimum:
        report["status"] = "insufficient_data"
    else:
        report["status"] = "ok"
        report["psi"] = {}
        for name, hist in reference["histograms"].items():
            counts = np.histogram([row[name] for row in rows], bins=hist["edges"])[0]
            value = psi(hist["counts"], counts)
            report["psi"][name] = value
            if value > threshold:
                report["alerts"].append(f"{name}_distribution_shift")
        empty_rate = float(np.mean([row["empty_vector"] for row in rows]))
        report["empty_vector_rate"] = empty_rate
        if empty_rate > max(reference["empty_vector_rate"] + 0.1, 0.2):
            report["alerts"].append("high_empty_vector_rate")
        report["mean_confidence"] = float(np.mean([row["confidence"] for row in rows]))
        report["logged_latency_p95_ms"] = float(np.percentile([row["latency_ms"] for row in rows], 95))
        if report["alerts"]:
            report["status"] = "warning"
    write_json(STATE / "monitoring.json", report)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--minimum", type=int, default=100)
    parser.add_argument("--threshold", type=float, default=0.2)
    args = parser.parse_args()
    if args.minimum < 1 or args.limit < args.minimum or args.threshold <= 0:
        parser.error("Require limit >= minimum >= 1 and positive threshold")
    monitor(args.limit, args.minimum, args.threshold)
