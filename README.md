# Stroke Prediction API

A machine learning project that trains, evaluates, persists, and serves a binary stroke classifier through FastAPI. It combines a scikit-learn preprocessing pipeline, validation-based threshold selection, typed HTTP contracts, automated tests, and a Docker image.

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
```

Inference does not retrain a model, recalculate preprocessing statistics, or reload the artifact on each request.

## Repository layout

```text
stroke_prediction_system/
├── .github/workflows/ci.yml       # GitHub Actions test workflow
├── artifacts/
│   └── stroke_model.joblib        # Generated locally; excluded from Git
├── data/raw/
│   └── healthcare-dataset-stroke-data.csv  # Supplied locally; excluded from Git
├── notebooks/01_eda.ipynb         # Exploratory analysis
├── src/stroke_prediction/
│   ├── api/
│   │   ├── main.py               # Lifespan, routes, error handling
│   │   └── schemas.py            # Input and output contracts
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

Development dependencies additionally include pytest, httpx2, Matplotlib, ipykernel, and Jupyter. Preserve compatible Python and library versions when loading serialized models. The artifact does not currently record its dependency versions or dataset checksum.

## Dataset and preprocessing

Place the CSV at `data/raw/healthcare-dataset-stroke-data.csv`. The inspected file contains **5,110 records**: 4,861 with `stroke=0` and 249 with `stroke=1` (approximately 4.87% positives). It contains 201 missing BMI values and 1,544 records with `smoking_status="Unknown"`.

The CSV is excluded from Git. Dataset acquisition, provenance, and usage terms are not documented in the current repository; obtain the matching dataset and verify its terms before use or redistribution.

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

Readiness checks artifact presence; it does not perform an inference self-test.

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

## Docker

The artifact must exist locally before building. It is excluded from Git but intentionally included in the Docker build context. The original CSV is not needed inside the image.

```bash
docker build -t stroke-prediction-api:1.1 .
docker run --rm -p 127.0.0.1:8000:8000 stroke-prediction-api:1.1
```

Open [http://localhost:8000/docs](http://localhost:8000/docs) and check `/health` and `/ready`. This example publishes the port only on the local host.

The image installs runtime requirements, copies source and artifact, and starts Uvicorn on container port 8000 as the non-root `app` user. That user can read the source and model. The image does not train or evaluate models. Verify its user with:

```bash
docker run --rm stroke-prediction-api:1.1 id
```

The base image, `python:3.13-slim`, is not pinned to a patch version or immutable digest. Builds may change when the base image changes. A missing artifact causes the Dockerfile's explicit `COPY` step to fail.

## Tests and CI

```bash
python -m pytest -q
```

Tests cover splitting, preprocessing, metrics, threshold selection, final-evaluation calculations, persistence equivalence, inference ordering, input/output schemas, lifespan behavior, health/readiness, and prediction responses. They use synthetic data and mocked loading/inference and do not require the CSV or real artifact.

`pyproject.toml` configures `--basetemp=./.pytest_tmp`. Pytest clears this directory between runs; keep it exclusively for test files. If Windows permissions prevent access, specify a fresh directory under a writable temporary location:

```powershell
$testTemp = Join-Path ([System.IO.Path]::GetTempPath()) ("stroke-tests-" + [guid]::NewGuid().ToString("N"))
python -m pytest -q --basetemp=$testTemp
```

During this review, **33 tests passed** with a fresh temporary directory. The existing default directory had a local permission issue. No model retraining or held-out test evaluation was performed.

GitHub Actions runs on pushes to `main` and pull requests targeting `main`. It uses `ubuntu-latest`, `actions/checkout@v7`, and `actions/setup-python@v7`, installs `requirements-dev.txt`, and executes `pytest -q`. The workflow does not currently build/publish Docker images, deploy the API, or evaluate the real artifact.

## Limitations

- **Generalization:** evidence is limited to one stratified random split of one dataset. External, temporal, and subgroup validation have not been established. With 50 positives in the test set, each correctly identified case changes recall by two percentage points.
- **Selection bias:** choosing the model and threshold on validation data can produce optimistic validation estimates. Recall of at least 80% on that subset is not a guarantee for future inputs. Do not adjust the threshold against the final test results.
- **Error burden:** the recorded test result includes 230 false positives and 10 false negatives. These results do not establish suitability for screening or clinical decisions.
- **Calibration and target:** no score calibration is implemented. Balanced class weights alter the training objective, and the dataset label does not establish a prospective prediction horizon.
- **Schema scope:** basic numeric limits and category names are checked, but cross-field consistency, measurement units, comprehensive plausibility limits, and distribution drift are not validated. Default Pydantic coercion remains enabled; numeric typing is not strict.
- **Missingness:** only BMI is imputed. Other missing fields are rejected at the API boundary; missingness patterns are not explicitly modeled.
- **Artifact management:** no signatures, checksum verification, dependency manifest, unique training-run identifiers, or model registry are implemented. Required-key checks do not make untrusted joblib files safe.
- **Deployment:** the HTTP API accepts one record per request and has no authentication, authorization, rate limiting, TLS termination, automatic model reload, or drift monitoring. Additional controls are needed before exposing it outside a controlled environment.
- **Privacy:** the generic response-error log is only one safeguard. Handling health-related inputs requires appropriate access controls and logging policies across clients, application, and infrastructure.

## License and data attribution

No software license file or dataset source/license declaration is currently included. Add an appropriate project license and verified dataset attribution before distributing the project or its data artifacts.
