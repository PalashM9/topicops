import hashlib

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import api, store
from app.common import sha256, write_json
from app.data import validate
from app.monitor import psi
from app.train import evaluate, new_pipeline, reference_profile


@pytest.fixture
def fixture_model():
    texts = [
        "graphics render image pixel display", "image graphics rendering display pixel",
        "cars engine vehicle driving road", "engine cars road driving vehicle",
        "baseball bat pitcher team game", "pitcher baseball game bat team",
        "space orbit nasa rocket moon", "rocket space moon nasa orbit",
    ]
    labels = ["comp.graphics"] * 2 + ["rec.autos"] * 2 + ["rec.sport.baseball"] * 2 + ["sci.space"] * 2
    model = new_pipeline()
    model.fit(texts, labels)
    return model, texts, labels


def test_training_and_profile(fixture_model):
    model, texts, labels = fixture_model
    metrics = evaluate(model, [{"text": text, "label": label} for text, label in zip(texts, labels)])
    assert metrics["macro_f1"] == 1.0
    profile = reference_profile(model, texts)
    assert sum(profile["histograms"]["confidence"]["counts"]) == len(texts)
    assert np.allclose(model.predict_proba(texts).sum(axis=1), 1)


def test_distribution_metric():
    assert psi([10, 20, 30], [10, 20, 30]) == pytest.approx(0)
    assert psi([100, 0], [0, 100]) > 0.2


def test_validation():
    with pytest.raises(ValueError):
        validate([])
    rows = [{"text": name, "id": hashlib.sha256(name.encode()).hexdigest(), "label": name}
            for name in ["comp.graphics", "rec.autos", "rec.sport.baseball", "sci.space"]]
    validate(rows)
    with pytest.raises(ValueError):
        validate(rows + [rows[0]])


def test_api(tmp_path, monkeypatch, fixture_model):
    model, _, _ = fixture_model
    models = tmp_path / "models"
    folder = models / "test-version"
    folder.mkdir(parents=True)
    joblib.dump(model, folder / "model.joblib")
    write_json(models / "production.json", {
        "version": "test-version", "model_sha256": sha256(folder / "model.joblib")
    })
    monkeypatch.setattr(api, "MODELS", models)
    monkeypatch.setattr(store, "DB", tmp_path / "predictions.sqlite")
    with TestClient(api.app) as client:
        assert client.get("/ready").status_code == 200
        result = client.post("/predict", json={"text": "nasa rocket moon orbit"})
        assert result.status_code == 200
        assert result.json()["label"] == "sci.space"
        assert result.json()["logging_ok"] is True
        assert client.post("/predict", json={"text": "   "}).status_code == 422
        assert client.post("/predict", json={"text": "x" * 10001}).status_code == 422
        assert "router_predictions_total" in client.get("/metrics").text
    rows = store.recent("test-version", 10)
    assert len(rows) == 1
    assert "text" not in rows[0]


def test_tampered_model_rejected(tmp_path, monkeypatch):
    models = tmp_path / "models"
    folder = models / "bad-version"
    folder.mkdir(parents=True)
    (folder / "model.joblib").write_bytes(b"bad artifact")
    write_json(models / "production.json", {"version": "bad-version", "model_sha256": "wrong"})
    monkeypatch.setattr(api, "MODELS", models)
    with pytest.raises(RuntimeError, match="integrity"):
        with TestClient(api.app):
            pass
