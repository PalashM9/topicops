# TopicOps: local-first MLOps for an NLP topic router

A complete portfolio starter: public text dataset, validation, reproducible training,
experiment tracking, candidate selection, guarded promotion, versioned artifacts,
FastAPI serving, privacy-conscious prediction metadata, drift heuristics, tests,
Docker deployment, Prometheus metrics, and GitHub Actions CI.

## Scope and cost

This is a four-topic classifier, not a trained customer-support intent classifier.
The classes are comp.graphics, rec.autos, rec.sport.baseball, and sci.space. The routing
architecture could later be adapted to real support intents with properly licensed,
labeled data. The dataset contains historical newsgroup messages, which may include
sensitive or offensive text. Do not publish raw data or logs.

No paid API, GPU, cloud subscription, or model-hosting account is needed. Your own
computer runs everything; electricity, hardware and internet are not literally free.
Python dependencies and initial dataset/container downloads require internet.
GitHub CI is optional: use a public repository with standard runners to avoid
private-repository allowance issues. Do not enable paid runners or services.
Use local Docker Engine or an alternative appropriate to your OS; Docker Desktop
licensing depends on your situation. You can skip Docker entirely.

## Quick start (Python 3.11)

Unzip and open a terminal inside the repository.

Linux/macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If activation is restricted, call `.venv\Scripts\python.exe` directly instead
of changing your machine's execution policy.

From the repository root:

```bash
ruff check .
pytest -q
python -m app.data
python -m app.train
python -m app.promote
uvicorn app.api:app --host 127.0.0.1 --port 8000
```

The first data run downloads 20 Newsgroups. Training is CPU-only. Promotion requires
validation macro F1 >= 0.70 and no >0.01 regression against the current production
model. If rejected, inspect validation.json; do not blindly lower the gate.
The API deliberately refuses to start without a promoted model.

Open http://127.0.0.1:8000/docs for the interactive prediction form.
Health: /health. Readiness with version: /ready. Prometheus: /metrics.

A separate terminal, with the same virtual environment activated:

```bash
python scripts/demo_traffic.py --count 200
python -m app.monitor --limit 200
```

Linux/macOS prediction example:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"NASA launched a rocket into orbit around the moon."}'
```

Windows PowerShell:

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8000/predict -Method Post -ContentType application/json -Body '{"text":"NASA launched a rocket into orbit around the moon."}'
```

## MLflow UI

Training writes to `state/mlflow.db`; MLflow artifacts are stored locally too.
Run from the same repository root:

```bash
mlflow ui --backend-store-uri sqlite:///state/mlflow.db --host 127.0.0.1 --port 5000
```

Open http://127.0.0.1:5000. Compare the three runs' validation macro F1 and accuracy.
Each run includes the serialized pipeline, metadata, validation report, and reference
profile. MLflow 2.20.1 is deliberately pinned here; upgrading should be a separate
change with tests and dependency review. These are fixed versions, not a claim that
they are the newest packages or a complete transitive lockfile.

## Architecture and data discipline

```
20 Newsgroups -> clean/deduplicate -> train / validation / test + SHA-256 manifest
train -> TF-IDF + LogisticRegression (C = 0.5, 1, 2) -> MLflow + versioned candidates
validation -> select candidate -> validation gate -> report test -> production pointer
production pointer -> FastAPI -> metadata SQLite + Prometheus counters/histogram
metadata SQLite + validation reference -> monitoring report -> human review
```

- Official train and test partitions stay separate. Exact duplicate texts across
  partitions are removed, giving priority to the training partition.
- Train is split into training and validation with a fixed seed and stratification.
- Headers, signatures, and quoted text are removed heuristically; this does not
  eliminate every potential source of leakage.
- Vocabulary is learned only on training. Test does not drive model selection or
  the promotion gate. Test results are cached for each candidate to discourage
  repeated evaluation. Inspecting test results and tuning afterward would still
  leak test information; maintain a fresh holdout for further iterations.
- Split fingerprints are verified before training. Model fingerprints are verified
  at promotion and serving. Hashes detect accidental changes, not malicious actors
  who can replace both the artifact and its checksum.
- Candidate directories use MLflow run IDs. production.json is an atomic pointer
  replacement on the same filesystem. Run only one training/promotion process
  at a time; concurrent administrative writes are not coordinated.
- Retraining currently uses the same dataset. Real retraining needs curated new
  labels and a new split/versioning plan, not unlabeled prediction traffic.

## Docker and Prometheus (optional)

Train and promote locally first. Stop the locally running API before starting Compose.

```bash
docker compose up --build -d
docker compose ps
docker compose logs api
```

API: http://127.0.0.1:8000/docs. Prometheus: http://127.0.0.1:9090.
Everything binds to localhost on the host. The API runs as a non-root container user.
On Linux the mounted state directory must be writable by the container's worker user
(typically UID 1000). If it is not, prefer local serving or configure a matching UID
in the image; do not make the directory world-writable.

Useful PromQL:

```promql
sum(rate(router_predictions_total[5m]))
sum by (label) (rate(router_predictions_total[5m]))
histogram_quantile(0.95, sum by (le) (rate(router_prediction_seconds_bucket[5m])))
router_logging_failures_total
```

The container image contains application code only. Trained state is mounted, not
baked into the image. Model serialization requires compatible dependency versions.
This setup uses one Uvicorn worker; do not scale workers without configuring
Prometheus multiprocess collection and reviewing SQLite write contention.

## Drift demonstration

```bash
python scripts/demo_traffic.py --count 200 --shifted
python -m app.monitor --limit 200
```

Reports are saved in state/monitoring.json. At least 100 recent predictions for the
currently promoted version are required by default. Reference distributions come
from validation data; live predictions supply length, max probability and whether
TF-IDF produced an empty feature vector. PSI uses fixed bins and additive smoothing.
The default 0.2 threshold is a demo heuristic, not a statistically calibrated test.
An empty-vector rate above max(reference + 0.1, 0.2) also produces a warning.
Small/repetitive demo traffic may trigger warnings even without the shifted option.

Monitoring does not measure live accuracy or prove concept drift: it has no ground
truth. A shift warning calls for label collection and investigation, not automatic
retraining or automatic promotion. Confidence is uncalibrated max class probability,
not a guarantee of correctness. The service always chooses a known class; it has no
out-of-domain rejection policy.

No raw request text is stored. SQLite stores timestamp, model version, predicted
label, confidence, length, empty-vector flag and an internal processing latency.
That latency excludes the log write and network transit. Prometheus latency includes
the log-write attempt but not network time. Metadata is still data: decide retention
before real use. Training data itself remains on local disk.

## Retrain and rollback

```bash
python -m app.train
python -m app.promote
# Restart local Uvicorn, or:
docker compose restart api

# Restore previous pointer, then restart serving:
python -m app.promote --rollback
docker compose restart api
```

Rollback toggles current and previous pointers. The API loads a model at startup;
it does not hot-reload after promotion. Re-promoting a candidate rewrites the previous
pointer, so do not use this as a full release-history manager. Version directories
remain available for manual inspection.

## CI and deployment boundary

Push this source code to a public GitHub repository to run .github/workflows/ci.yml.
It runs linting, offline synthetic-data tests and a Docker image build. The test suite
covers training/probabilities, reference histograms, PSI, data validation, API input
validation, metadata logging and artifact integrity. CI intentionally does not download
the dataset or train the real model. No credentials or model files are committed.

This is CI plus manual, gated local delivery; it does not claim automated remote CD.
A production service would additionally need authentication, TLS, rate limiting,
backups, dependency scanning, labeled performance monitoring, lockfiles, orchestration,
retention policies and a governed promotion workflow. Never expose this demo API or
MLflow UI directly to the internet. Only load joblib files you created and trust;
loading serialized Python objects can execute code.

## Repository layout

```
app/                  data, training, promotion, API, storage, monitoring
scripts/demo_traffic.py
monitoring/prometheus.yml
tests/test_project.py
.github/workflows/ci.yml
Dockerfile
compose.yaml
requirements.txt
pyproject.toml
state/                generated, gitignored (data, models, tracking, prediction logs)
```

## Troubleshooting

- Download fails: check connectivity and retry python -m app.data. Do not change
  train/test splitting to work around a network error.
- No production.json: finish training and a successful promotion before serving.
- Port in use: stop the other API/Compose service, or choose a different host port.
- Module not found: run commands from the repository root with the virtual environment.
- Promotion rejected: review validation confusion matrix and class reports; no score
  has been assumed or pre-filled in this project.
- Drift has insufficient data: generate >=100 requests for the current model version.
- State disappears: do not delete state; keep it mounted for container serving.

## Verification of the supplied archive

All Python files were syntax-checked during generation. End-to-end installation,
real-dataset training, network downloads and Docker execution were not performed in
the generation environment. Run pytest, the training sequence, and Compose locally
before treating this as a validated deployment.

## Public documentation used for project design

- https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html
- https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html
- https://mlflow.org/docs/latest/ml/tracking/tutorials/local-database/
- https://docs.github.com/en/billing/concepts/product-billing/github-actions
