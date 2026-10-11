import hashlib
import json
import os
from pathlib import Path

ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
STATE = ROOT / "state"
DATA = STATE / "data"
MODELS = STATE / "models"
CATEGORIES = ["comp.graphics", "rec.autos", "rec.sport.baseball", "sci.space"]
SEED = 42


def ensure_dirs():
    for folder in [STATE, DATA, MODELS]:
        folder.mkdir(parents=True, exist_ok=True)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
