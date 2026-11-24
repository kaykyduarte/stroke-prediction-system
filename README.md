# Stroke Prediction API

A machine learning project that trains, evaluates, persists, and serves a binary stroke classifier through FastAPI. It combines a scikit-learn preprocessing pipeline, validation-based threshold selection, and typed HTTP contracts with HTTP metrics and latency logging, window-based alerts with hysteresis, supervised asynchronous tasks, bounded automatic monitor recovery, and Docker smoke tests in CI.

**Intended use:** educational experimentation and software engineering practice. The model predicts the dataset's `stroke` label. It is not a diagnosis, a validated clinical decision tool, or a calibrated estimate of an individual's future stroke risk.

## Architecture

```text
Training CSV
  -> stratified train / validation / test split
  -> fit preprocessing and classifiers on training data
  -> compare models and select thresholds on validation data
  -> export the selected pipeline and full-precision threshold

API startup
  -> load the trusted artifact once per server process
  -> store it in app.state.artefato

POST /predict
  -> StrokeInput validation
  -> single-row DataFrame
  -> fitted pipeline.predict_proba()
  -> apply the stored threshold
  -> StrokeOutput validation
  -> JSON response

Monitoring and automatic recovery
  -> HTTP responses update cumulative counters and a 300-second window
  -> every 30 seconds, evaluate the window and apply alert hysteresis
  -> record evaluation health and confirm stability after sustained successes
  -> supervise unexpected monitor termination
  -> authorize a bounded restart and await exponential backoff
  -> start a new supervised monitor with a fresh stability period

Shutdown
  -> mark the application as stopping; reject new recovery scheduling
  -> cancel and await pending recovery, then the current monitor
  -> release the model reference
```

Inference does not retrain a model, recalculate preprocessing statistics, or reload the artifact on each request. HTTP alerts, evaluation health, and task termination are separate signals: an elevated HTTP error rate does not itself restart the monitor. Automatic recovery addresses a terminated monitor task, not the prediction model or failed client requests.

## Repository layout

```text
stroke_prediction_system/
├── .github/workflows/ci.yml       # Tests, Docker build, and HTTP smoke tests
├── artifacts/
│   └── stroke_model.joblib        # Generated locally; excluded from Git
├── data/raw/
│   └── healthcare-dataset-stroke-data.csv  # Supplied locally; excluded from Git
├── notebooks/01_eda.ipynb         # Exploratory analysis
├── scripts/
│   └── generate_ci_artifact.py   # Synthetic artifact for container tests
├── src/stroke_prediction/
│   ├── api/
│   │   ├── main.py               # Lifespan, routes, task factories and callbacks
│   │   ├── schemas.py            # Input and output contracts
│   │   ├── security.py           # Internal-endpoint API-key validation
│   │   ├── metrics.py            # Cumulative HTTP response counters
│   │   ├── window_metrics.py     # Per-second buckets in a rolling window
│   │   ├── metrics_summary.py    # Error percentages and summary helpers
│   │   ├── alert_evaluation.py   # Minimum-volume gate and alert evaluation
│   │   ├── alert_state.py        # Incident transitions and hysteresis
│   │   ├── alert_monitor.py      # Periodic evaluation loop
│   │   ├── monitor_health.py     # Degradation, inactivity, and stability
│   │   ├── alert_supervision.py  # Monitor termination logging
│   │   ├── retry_policy.py      # Exponential backoff calculation
│   │   ├── retry_state.py       # Thread-safe restart budget
│   │   ├── restart_planning.py  # Restart authorization and delay planning
│   │   ├── restart_executor.py  # Cancellable backoff and task creation
│   │   ├── restart_recovery.py  # Recovery orchestration
│   │   └── recovery_policy.py   # Reset restart budget after stability
│   ├── data/preprocessing.py     # Splitting, imputation, encoding, scaling
│   └── models/
│       ├── train.py              # Experiments and artifact export
│       ├── evaluate.py           # Metrics and threshold selection
│       ├── final_evaluation.py   # Evaluation of an already trained pipeline
│       ├── persistence.py        # Artifact saving and loading
│       └── predict.py            # Inference and classification
├── tests/
├── Dockerfile
├── .dockerignore
├── pyproject.toml                # Pytest configuration
├── requirements.txt             # Runtime dependencies
└── requirements-dev.txt         # Runtime, test, and notebook dependencies
```

## Setup

Use Python **3.13**. The inspected local environment uses Python **3.13.14**. Run commands from the repository root: the current imports use the `src.stroke_prediction` module path.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

Linux/macOS:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

For API execution without development tooling, install `requirements.txt` instead. The development file includes it through `-r requirements.txt`.

The runtime requirements currently pin:

| Dependency | Version |
| --- | --- |
| FastAPI | 0.162.2 |
| Uvicorn | 0.54.0 |
| Pydantic | 2.14.2 |
| pandas | 3.0.6 |
| NumPy | 2.5.4 |
| scikit-learn | 1.9.1 |
| SciPy | 1.18.1 |
| joblib | 1.6.0 |
| threadpoolctl | 3.7.0 |
| Starlette | 1.10.0 |

Development dependencies additionally pin pytest 9.1.1, **httpx2 2.16.0**, Matplotlib 3.11.2, ipykernel 7.4.0, and Jupyter 1.1.1. `httpx2` is intentional: the installed Starlette 1.10.0 test client imports it; it is not a typo for `httpx`. Preserve compatible Python and library versions when loading serialized models. The artifact does not currently record its dependency versions or dataset checksum.

### Internal monitor API key

`MONITOR_API_KEY` configures the expected key for `GET /internal/monitor-health`. Clients supply it through the `X-Monitor-Key` header. Missing configuration denies access; it does not disable authentication or prevent the public API from starting.

For a local demonstration only, set a fictitious value before starting Uvicorn:

```powershell
$env:MONITOR_API_KEY = "chave-local-teste"
```

```bash
export MONITOR_API_KEY='chave-local-teste'
```

This published example is not a production secret. Use an independently generated secret through your deployment's secret-management mechanism, never commit real keys, and use HTTPS for network access. The current application reads the process environment; it does not automatically load a `.env` file. The key is not required for `/health`, `/ready`, or `/predict`.

## Dataset and preprocessing

Place the CSV at `data/raw/healthcare-dataset-stroke-data.csv`. The inspected file contains **5,110 records**: 4,861 with `stroke=0` and 249 with `stroke=1` (approximately 4.87% positives). It contains 201 missing BMI values and 1,544 records with `smoking_status="Unknown"`.

The dataset source supplied for this project is [Stroke Prediction Dataset on Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset), published under the account `fedesoriano`.

To prepare the training data:

1. Download the dataset from the linked Kaggle page.
2. Extract `healthcare-dataset-stroke-data.csv` from the download.
3. Place it in `data/raw/` without changing the filename or record ordering.

The CSV is excluded from Git. The Kaggle source is now identified, but this repository does not independently establish the original collection provenance or clinical representativeness of the records. Verify the source page's current license and usage terms before use or redistribution.

`id` and `stroke` are excluded from input features. The shared `ColumnTransformer` applies:

| Feature group | Transformation |
| --- | --- |
| `bmi` | Training-set median imputation, then `StandardScaler` |
| `age`, `avg_glucose_level` | `StandardScaler` |
| Five categorical features | `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` |
| `hypertension`, `heart_disease` | Preserve binary values |

All preprocessing parameters are learned exclusively from training data. `Unknown` is retained as a legitimate smoking-status category. Only BMI is imputed; other features are expected to be present and valid.

The encoder tolerates categories absent from its training subset. Separately, the API restricts categories to the dataset contract: an API-valid category can still have been absent from the training subset.

## Training and export

After placing the CSV in the expected directory, run:

```bash
python -m src.stroke_prediction.models.train
```

Two stratified splits use `random_state=42`:

| Partition | Share | Records | Purpose |
| --- | ---: | ---: | --- |
| Training | 60% | 3,066 | Fit preprocessing and classifiers |
| Validation | 20% | 1,022 | Compare models and select thresholds |
| Test | 20% | 1,022 | Final held-out evaluation |

The experiment compares:

- A majority-class `DummyClassifier(strategy="most_frequent")`.
- Model A: `LogisticRegression(max_iter=1000)`.
- Model B: `LogisticRegression(max_iter=1000, class_weight="balanced")`.

Each model has an independent preprocessing instance. Validation analysis uses fixed thresholds of 0.50, 0.30, 0.20, and 0.10, along with Average Precision. Threshold selection maximizes precision subject to recall being at least 80%; ties prefer higher recall and then a higher threshold.

The export configuration is fixed to **Model B**. Its previously selected threshold is **`0.52035772224751`**. Classification uses `score >= threshold`; the full value is stored.

The normal training command fits the configured models, recalculates validation thresholds, and exports Model B to `artifacts/stroke_model.joblib`, overwriting an existing file. It does not evaluate the held-out test set: `main(avaliar_teste=False)` is the default. There is no command-line flag for final evaluation. Do not enable test evaluation during routine export or tune choices against test results.

Reproducing the same split requires the same records and ordering. Numerical reproducibility also depends on the software environment.

### Artifact contents

| Key | Contents |
| --- | --- |
| `pipeline` | Fitted preprocessing and classifier |
| `limiar` | Full-precision classification threshold |
| `versao_artefato` | Artifact format version, currently `"1.0"` |

The version identifies the format, not a unique training run. The loader checks the dictionary and required keys, but does not comprehensively validate value types, schema compatibility, or provenance. Only load trusted project-generated files: joblib deserialization can execute code.

## Recorded evaluation

These results come from the previously recorded experiment and final evaluation. The test set was not reevaluated during this documentation update.

| Metric | Validation | Held-out test |
| --- | ---: | ---: |
| Accuracy | 76.52% | 76.52% |
| Precision, class 1 | 15.07% | 14.81% |
| Recall, class 1 | 82.00% | 80.00% |
| Average Precision | 16.63% | 27.42% |
| True negatives | 741 | 742 |
| False positives | 231 | 230 |
| False negatives | 9 | 10 |
| True positives | 41 | 40 |

The test confusion matrix has actual classes as rows and predicted classes as columns, ordered `[0, 1]`:

```text
                    Predicted 0   Predicted 1
Actual 0                742           230
Actual 1                 10            40
```

At the fixed threshold, the model identified 40 of 50 positive test records and classified 230 negative records as positive. Only 40 of 270 positive predictions were correct. Average Precision summarizes ranking across thresholds; it is not precision at the deployed threshold.

The majority-class baseline achieved 95.11% validation accuracy with 0% recall, illustrating why accuracy alone is insufficient for this imbalanced task. The selected configuration accepts a substantial false-positive burden to improve recall. The model and threshold were not changed after observing final test results.

## Run the API

Generate the artifact or obtain a trusted, compatible project artifact, then start:

```bash
python -m uvicorn src.stroke_prediction.api.main:app --host 127.0.0.1 --port 8000
```

Open [Swagger UI](http://127.0.0.1:8000/docs). The OpenAPI schema is available at `/openapi.json`.

During startup, `lifespan` loads `artifacts/stroke_model.joblib` relative to the project root and stores it in `app.state.artefato`. Each process loads its own copy. A missing file is logged and leaves the app running without a model; other loading errors propagate and can prevent startup. Replacing the artifact requires restarting the process.

| Endpoint | Behavior |
| --- | --- |
| `GET /health` | HTTP 200 with `{"status":"ok"}`, independent of the model once the app has started |
| `GET /ready` | HTTP 200 with `{"status":"ready"}` if an artifact is present; otherwise HTTP 503 with `{"status":"not_ready"}` |
| `POST /predict` | Validate one record and return score, classification, and artifact version |
| `GET /internal/monitor-health` | Return monitor state after `X-Monitor-Key` authentication; excluded from public OpenAPI |

Readiness checks artifact presence; it does not perform an inference self-test.

### Protected monitor health

Send the configured key in the request header:

```bash
curl -H "X-Monitor-Key: chave-local-teste" http://127.0.0.1:8000/internal/monitor-health
```

| Condition | HTTP status | Response |
| --- | ---: | --- |
| `MONITOR_API_KEY` is not configured or is empty | 503 | `{"detail":"Acesso ao monitor indisponível."}` |
| Header is missing or incorrect | 403 | `{"detail":"Acesso negado."}` |
| Correct key | 200 | Current monitor state |

An illustrative healthy response is:

```json
{
  "falhas_consecutivas": 0,
  "sucessos_consecutivos": 2,
  "degradado": false,
  "inativo": false,
  "tarefa_ativa": true
}
```

| Field | Meaning |
| --- | --- |
| `falhas_consecutivas` | Consecutive metric-evaluation failures |
| `sucessos_consecutivos` | Consecutive successful evaluations in the current sequence |
| `degradado` | Evaluation health is degraded; recovery has not yet been confirmed |
| `inativo` | More than 90 seconds have elapsed since the last recorded evaluation or execution-start reference |
| `tarefa_ativa` | The current monitor task has not completed, failed, or been cancelled |

The values are dynamic. HTTP 200 means the authorized state query succeeded, even if the monitor is degraded, inactive, or terminated. These flags do not change `/ready`: a loaded model can remain available while monitoring is impaired. A recently terminated task can report `tarefa_ativa=false` and `inativo=false` simultaneously.

The route uses `include_in_schema=False`, so it is absent from `/docs` and the public OpenAPI schema. This is not a security boundary; the dependency validates the header using `secrets.compare_digest`. API keys and request bodies are not intentionally included in the monitoring logs.

### Input contract

All ten fields are required, including `bmi`, which may explicitly be `null`. Extra fields, including `id` and `stroke`, are rejected. Category spelling and capitalization must match exactly.

| Field | Accepted values |
| --- | --- |
| `gender` | `Male`, `Female`, `Other` |
| `age` | Number from 0 to 125, inclusive |
| `hypertension` | `0` or `1` |
| `heart_disease` | `0` or `1` |
| `ever_married` | `Yes`, `No` |
| `work_type` | `Private`, `Self-employed`, `Govt_job`, `children`, `Never_worked` |
| `Residence_type` | `Urban`, `Rural` |
| `avg_glucose_level` | Number greater than 0 |
| `bmi` | Number greater than 0, or `null` |
| `smoking_status` | `formerly smoked`, `never smoked`, `smokes`, `Unknown` |

Example request body:

```json
{
  "gender": "Female",
  "age": 65,
  "hypertension": 1,
  "heart_disease": 0,
  "ever_married": "Yes",
  "work_type": "Private",
  "Residence_type": "Urban",
  "avg_glucose_level": 180.5,
  "bmi": null,
  "smoking_status": "formerly smoked"
}
```

Send it through `/docs`, or save it as `request.json` and run:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" --data-binary @request.json
```

In Windows PowerShell, use `curl.exe` if `curl` resolves to a PowerShell alias.

### Output contract and errors

This response illustrates the format only; it is not a claimed prediction for the example request:

```json
{
  "score": 0.6,
  "classificacao": 1,
  "versao_artefato": "1.0"
}
```

`score` must be in `[0, 1]`, `classificacao` must be `0` or `1`, and `versao_artefato` is a required string. The score is the balanced classifier's positive-class output, not a calibrated clinical probability.

| Status | Meaning |
| --- | --- |
| 200 | Valid input, available model, and valid inference response |
| 422 | Invalid input, such as an unknown category, missing field, or invalid BMI |
| 503 | No model available; prediction returns `{"detail":"Modelo indisponível."}` |
| 500 | Response schema violation; returns `{"detail":"Erro interno na resposta do modelo."}` |

The response-validation handler logs a generic message without request contents or exception values. Standard request-validation responses may echo invalid input; clients and infrastructure should avoid unnecessary sensitive-payload logging. The custom handler does not cover every possible inference/runtime exception.

## Observability and monitoring

The HTTP middleware logs method, registered route template, response status, and duration in milliseconds using `perf_counter()`. Unmatched paths are labeled `unmatched`; arbitrary URL values, query strings, and request bodies are not included in this log. Application logging must be configured to emit `INFO` messages to make these latency records visible. Duration covers processing until the response returns to the middleware, not complete transmission to the client.

Each response returned through the middleware updates two independent in-memory collectors:

- `MetricasAPI`: cumulative total and separate 2xx, 4xx, and 5xx counts since startup. Other status classes count toward the total only.
- `MetricasJanela`: total and 5xx counts over the last 300 seconds, using chronological `deque` buckets with one-second resolution and a monotonic clock. Both registration and queries expire old buckets, including after idle periods.

Collectors use locks and return copies of their counters. The middleware counts public endpoints, internal endpoints, and unmatched-route responses alike. Probe traffic and authentication-related responses therefore also affect the measured traffic and error rate. Unhandled exceptions that escape `call_next()` before a response is returned are not recorded by the current middleware.

Summary helpers calculate `5xx / total * 100`; no traffic produces `None`, not a zero-error estimate. They are reusable components, not a public `/metrics` endpoint. There is no Prometheus integration.

### HTTP incident policy

| Setting | Implemented value |
| --- | --- |
| Rolling window | 300 seconds |
| Evaluation interval | 30 seconds; first evaluation occurs after the initial wait |
| Minimum volume for an alert-state decision | 20 responses in the window |
| Incident activation | 5xx rate **at least 10%** |
| Incident recovery | 5xx rate **at most 5%** |

Hysteresis preserves the existing incident state between 5% and 10%. Insufficient volume also preserves it: even a 0% error rate from fewer than 20 responses does not prove recovery. A warning is logged only when a new incident starts, avoiding repeated notifications for the same active incident. No external notification is sent.

These are educational thresholds, not production availability objectives. An insufficient sample does not establish health, and this statistical rule does not replace handling of individual critical failures.

### Health of the monitoring loop

This is a separate state machine from HTTP incident hysteresis:

| Signal | Policy |
| --- | --- |
| Degradation | Three or more consecutive evaluation failures |
| Recovery from degradation | Two consecutive successful evaluations |
| Inactivity | More than 90 seconds since the last evaluation/reference |
| Task termination | Observed independently through `asyncio.Task.done()` |

Both success and failure refresh the evaluation timestamp: an evaluation can fail while the monitor remains active. One success resets the consecutive-failure counter but does not immediately clear a previously degraded state. Any failure resets the consecutive-success sequence.

## Automatic recovery

The task factory creates the monitor, updates the shared task reference, and attaches a termination callback. Unexpected exceptions are logged with their original traceback; a normal return is also unexpected for this continuous loop. Expected cancellation does not trigger recovery.

When a non-cancelled monitor ends, recovery scheduling first checks that shutdown has not begun and that another recovery is not already pending. The planner atomically reserves a restart attempt through `ControleReinicializacoes`, and the executor awaits the backoff before preparing a new execution and calling the same supervised task factory.

| Setting | Implemented value |
| --- | --- |
| Maximum authorized restarts before a stability reset | 3 |
| Attempt 1 delay | 1 second |
| Attempt 2 delay | 2 seconds |
| Attempt 3 delay | 4 seconds |
| Generic backoff cap | 30 seconds; not reached with this three-attempt budget |
| Stable success period before resetting the budget | At least 120 seconds |
| Minimum consecutive successes | At least 2 |
| Maximum age of the latest evaluation for stability | 90 seconds |

The same restart-control instance is passed to successive monitors. Creating a new task does not reset its budget. Each new execution clears the success-period start and success count and refreshes the inactivity reference, while preserving degradation. Stability is confirmed only after sufficient successes over the required period, with a recent evaluation. A failure or a gap of more than 90 seconds between evaluations restarts the success period. With regular 30-second evaluations, five successes spanning 120 seconds satisfy the time criterion.

Evaluation failures are logged and counted without terminating the loop. A failure in the separate recovery-confirmation policy is logged independently and does **not** turn a successful metric evaluation into a failure. Cancellation remains propagatable. If the recovery task itself fails, its dedicated callback records that exception; it does not recursively recover the recovery task.

If all three restart attempts are exhausted and the last monitor ends, there is no automatic fifth monitor execution. With no monitor left to produce successful evaluations, the stability policy cannot reset the budget. Operator intervention or a new process startup may be necessary. `/health` can still return 200, `/ready` can remain 200 while the artifact is loaded, and the protected endpoint reports `tarefa_ativa=false`.

Shutdown sets `app.state.encerrando=True` before cancelling and awaiting recovery. It then obtains, cancels, and awaits the current monitor and releases the artifact reference. New recoveries are blocked during shutdown. Supervision callbacks own exception logging; shutdown handles already observed task exceptions without duplicating those logs.

The pending-recovery guard prevents concurrent recovery tasks in the same event loop; it is not a durable record that deduplicates every repeated termination notification. All counters, incident states, health references, and restart budgets are per-process and are lost on restart. Run a **single worker** for the current design: additional workers each create their own monitor and can generate duplicate alerts.

## Docker

The artifact must exist locally before building. It is excluded from Git but intentionally included in the Docker build context. The original CSV is not needed inside the image.

```bash
docker build -t stroke-prediction-system:local .
docker run --rm --name stroke-api -p 127.0.0.1:8000:8000 -e MONITOR_API_KEY=chave-local-teste stroke-prediction-system:local
```

Open [http://localhost:8000/docs](http://localhost:8000/docs). From another terminal, check the three operational endpoints:

```bash
curl --fail http://127.0.0.1:8000/health
curl --fail http://127.0.0.1:8000/ready
curl --fail -H "X-Monitor-Key: chave-local-teste" http://127.0.0.1:8000/internal/monitor-health
```

On Windows PowerShell, use `curl.exe`. This example publishes the port only on the local host and uses a deliberately fictitious demonstration key. Do not reuse that key outside local testing or bake a real secret into an image. An actual deployment should inject its own key, for example by forwarding an already configured environment variable with `-e MONITOR_API_KEY`, and provide HTTPS for network access.

The image installs runtime requirements, copies source and artifact, and starts Uvicorn on container port 8000 as the non-root `app` user. That user can read the source and model. The image does not train or evaluate models. Verify its user with:

```bash
docker run --rm stroke-prediction-system:local id
```

The base image, `python:3.13-slim`, is not pinned to a patch version or immutable digest. Builds may change when the base image changes. A missing artifact causes the Dockerfile's explicit `COPY` step to fail.

## Tests and CI

```bash
python -m pytest -q
```

Tests cover splitting, preprocessing, model evaluation and threshold selection, persistence equivalence, inference ordering, and input/output schemas. Operational coverage includes API-key security, HTTP metrics, rolling-window expiration, alert hysteresis, concurrent restart-budget access, task supervision, exponential backoff, real cancellation during backoff, stability and inactivity, recovery failures, restart exhaustion, and coordinated shutdown. Tests use synthetic data and mocked loading/inference and do not require the original CSV or real model artifact.

`pyproject.toml` configures `--basetemp=./.pytest_tmp`. Pytest clears this directory between runs; keep it exclusively for test files. If Windows permissions prevent access, specify a fresh directory under a writable temporary location:

```powershell
$testTemp = Join-Path ([System.IO.Path]::GetTempPath()) ("stroke-tests-" + [guid]::NewGuid().ToString("N"))
python -m pytest -q --basetemp=$testTemp
```

Latest locally reported verification: **160 tests passed in 4.23 seconds**. The GitHub Actions workflow also
completed successfully. No real-model retraining or held-out test evaluation was performed during this documentation update.

### GitHub Actions pipeline

The workflow runs on pushes to `main` and pull requests targeting `main`. Both jobs use `ubuntu-latest`, `actions/checkout@v7`, and `actions/setup-python@v7` with Python 3.13.

```text
Push / pull request to main
  -> tests: install requirements-dev.txt -> python -m pytest -q
  -> docker-build (requires tests to pass):
       install requirements.txt
       -> generate a synthetic artifact
       -> build stroke-prediction-api:ci
       -> start the container
       -> GET /health and /ready with bounded retries
       -> POST /predict and validate status and JSON contract
       -> show container state/logs on failure
       -> remove the container with if: always()
```

The prediction smoke test requires HTTP 200, a numeric score in `[0, 1]`, an integer classification of 0 or 1, and artifact version `"1.0"`. It does not assert a particular score or predictive quality. The current Docker job does not configure or smoke-test the protected monitor endpoint.

The synthetic generator reuses the project's preprocessing and persistence, trains only on fictitious data, includes missing BMI, requires an explicit output path, and refuses existing destinations. For a local experiment, keep it separate from the real artifact:

```bash
python -m scripts.generate_ci_artifact --output artifacts/stroke_model_ci.joblib
```

The clean CI workspace instead generates `artifacts/stroke_model.joblib`, the path required by the Dockerfile. Never replace a real deployment artifact with the synthetic CI model. Containers are started without `--rm` so failure logs remain available until explicit cleanup. The workflow builds and tests locally on the runner; it **does not publish images to Docker Hub or another registry, deploy to a cloud service, or evaluate the real model**.

## Limitations

- **Generalization:** evidence is limited to one stratified random split of one dataset. External, temporal, and subgroup validation have not been established. With 50 positives in the test set, each correctly identified case changes recall by two percentage points.
- **Selection bias:** choosing the model and threshold on validation data can produce optimistic validation estimates. Recall of at least 80% on that subset is not a guarantee for future inputs. Do not adjust the threshold against the final test results.
- **Error burden:** the recorded test result includes 230 false positives and 10 false negatives. These results do not establish suitability for screening or clinical decisions.
- **Calibration and target:** no score calibration is implemented. Balanced class weights alter the training objective, and the dataset label does not establish a prospective prediction horizon.
- **Schema scope:** basic numeric limits and category names are checked, but cross-field consistency, measurement units, comprehensive plausibility limits, and distribution drift are not validated. Default Pydantic coercion remains enabled; numeric typing is not strict.
- **Missingness:** only BMI is imputed. Other missing fields are rejected at the API boundary; missingness patterns are not explicitly modeled.
- **Artifact management:** no signatures, checksum verification, dependency manifest, unique training-run identifiers, or model registry are implemented. Required-key checks do not make untrusted joblib files safe.
- **Access control:** `/predict`, `/health`, and `/ready` have no application-level authentication. `/internal/monitor-health` requires a shared API key, but there are no per-user roles, key rotation/auditing facilities, or rate limits. Hiding a route from OpenAPI does not make it private. Use appropriate network controls and HTTPS before exposing the service.
- **Observability scope:** HTTP availability and 5xx monitoring do not measure predictive accuracy, calibration, fairness, or clinical outcomes. There is no data-drift or model-drift detection, persistent external alert delivery, or durable incident store. Logging an alert does not guarantee that an operator receives it.
- **Process-local recovery:** metrics, alert state, monitor health, and restart budgets live only in memory within each process. Locks protect threads, not separate workers or containers. The current single-worker design can stop monitoring after restart exhaustion; no external watchdog or automatic process restart is implemented here.
- **Deployment:** the API accepts one record per HTTP request. TLS termination, automatic model reload, and a production deployment platform are not included. Container smoke tests use a synthetic model and do not establish production behavior or validate real-model compatibility across platforms.
- **Privacy:** the generic response-error log is only one safeguard. Handling health-related inputs requires appropriate access controls and logging policies across clients, application, and infrastructure.

## License and data attribution

Dataset attribution: **fedesoriano, Stroke Prediction Dataset, Kaggle**. Source: [kaggle.com/datasets/fedesoriano/stroke-prediction-dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).

The dataset license has not been independently verified against readable primary-source metadata during this documentation update. Consult the Kaggle listing for the applicable license and terms; source attribution alone does not grant redistribution rights.

No software license file is currently included in this repository. A dataset's license does not automatically license this project's source code. Add an appropriate project license before distributing the software under specific license terms.
