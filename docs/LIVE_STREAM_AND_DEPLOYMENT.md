# Live transaction prototype and deployment notes

## Event flow

`python -m demo.live_stream` is a sequential HTTP producer, not a score replay.
It obtains the next valid step from `/live/status`, then sends each raw event
to the authenticated `POST /score_transaction` route. The existing service
builds causal behavioural features, runs Model B and TreeSHAP, and writes the
scored result and transaction facts to the same local SQLite transaction
history. A failed feature, inference, explanation, or SQLite insert does not
produce a successful event record. Optional `event_id` values are unique in
the SQLite store and duplicates are rejected.

Each event gets a strictly increasing step. Existing state is resumed at
`max(reference_max_step, latest_online_step) + 1`; the immutable reference
Parquet is only read. The simulator has no fraud labels and never calculates
or stores scores itself.

The first four deterministic events use an initially unseen sender/receiver:

| Event | Input pattern | Intended evidence transition |
|---|---|---|
| 1 | small PAYMENT | New behavioural history |
| 2 | another small PAYMENT | One prior online event |
| 3 | CASH_OUT for 5,000 | Changed transaction behaviour |
| 4 | CASH_OUT for 1,000,000 | High Model B score at the current validation threshold |

The table describes the inputs, not predetermined output. Scores and flags
are recomputed by the checked-in Model B and explanations on each API call.
Later runs continue at later steps, so the initial New-history state applies
only when the account pair has not already appeared in stored history. Use a
separate clean state directory for a repeat of the initial sequence.

The live status, event feed, event-detail, and investigation-state endpoints
require the configured API key. `GET /live/events` is a persisted result ledger,
not a second scoring path. `GET /live/events/{event_key}` returns the selected
event, earlier online activity, and reference-network context;
`GET`/`PUT /live/events/{event_key}/investigation` reads or updates the
investigation status and analyst note stored with the event in SQLite. Updates
are audit-logged. The Streamlit LIVE MONITOR polls the feed every two seconds
while open and loads the selected event's risk assessment, explanation,
behavioural evidence, related activity, network context, and processing-stage
times from the API.

### Verified four-event scenario

Using the checked-in Model B bundle, reference maximum step 743, a clean
temporary state directory, and the unchanged 97.69/100 validation threshold,
the actual HTTP stream produced:

| Event | Step | History status | Risk score | Review result |
|---|---:|---|---:|---|
| 1 · PAYMENT 5 | 744 | New | 0.25/100 | Low; not flagged |
| 2 · PAYMENT 8 | 745 | Limited history | 0.28/100 | Low; not flagged |
| 3 · CASH_OUT 5,000 | 746 | Limited history | 97.37/100 | Below threshold; not flagged |
| 4 · CASH_OUT 1,000,000 | 747 | Limited history | 99.32/100 | High review priority; flagged |

These are the observed outputs of that local run, not values embedded in the
simulator. The dashboard displayed the event feed, selected a flagged event,
and rendered its TreeSHAP explanation and behavioural evidence.

## Run locally

The API, dashboard, and simulator use the same API key. In PowerShell, set a
random value in each process environment without putting it in a command-line
argument or source file:

```powershell
$env:SYNDICAI_API_KEY = (python -c "import secrets; print(secrets.token_urlsafe(32))")
python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

The primary institutional web frontend is served directly by the API at
`http://127.0.0.1:8000/` (loading `frontend/index.html`, `styles.css`, and
`app.js` via the `/static` mount).

Alternatively, the Streamlit monitoring desk remains available in a second terminal:

```powershell
$env:SYNDICAI_API_KEY = "<same locally-held value>"
$env:SYNDICAI_API_URL = "http://127.0.0.1:8000"
python -m streamlit run frontend/streamlit_app.py --server.port 8501
```

In a third terminal, set the same values and start the producer:

```powershell
$env:SYNDICAI_API_KEY = "<same locally-held value>"
$env:SYNDICAI_API_URL = "http://127.0.0.1:8000"
python -m demo.live_stream --interval 1.0
```

The stream runs until Ctrl+C. For a finite four-event demo use
`python -m demo.live_stream --interval 0.2 --max-events 4`. A custom
`--run-id` is useful for reproducibility, but event IDs are deduplicated;
choose a fresh ID prefix for each run against persistent state.

## Benchmark

Run the sequential API benchmark while the API is running:

```powershell
python -m demo.benchmark_live --events 50
```

The command prints a JSON report with feature-generation, XGBoost inference,
TreeSHAP explanation, state-write, scoring-work, and client-observed
end-to-end latency (median, p95, and mean), plus sequential events/second.
The report records CPU architecture, Python and core dependency versions,
sample count, run ID, and method. Its initial status request initializes the
API service before timing starts; it sends no scoring warm-up request and
includes the first scored event in the sample. It writes scored events to the
configured SQLite history; use an isolated copy of `models/` or set
`SYNDICAI_STATE_DIR` on the API to avoid mixing benchmark events into another
local run.

The API also returns an `X-Process-Time-Ms` header for request diagnostics.
The benchmark's client-side measurement includes HTTP round trip and response
handling and is the reported end-to-end measure. These local sequential
measurements are not a service-level guarantee or a production real-time
claim. The SQLite write stage is measured before commit; client-observed
end-to-end timing includes the full commit and response.

### Measurement recorded for this implementation

The benchmark was run on 2026-10-03 against the local FastAPI process after
service initialization and the SQLite indexed reference-history optimization.
It timed 50 consecutive Model B requests with the four-event demo already in
the isolated SQLite history. There was one producer and no intentional delay.
API startup was excluded; there were no scoring warm-up requests. End-to-end
is measured around each complete HTTP request.

| Measurement | Median | P95 | Mean |
|---|---:|---:|---:|
| Behavioural feature generation (SQLite indexed) | 23.4 ms | 32.1 ms | 24.1 ms |
| Model inference | 2.3 ms | 3.6 ms | 2.5 ms |
| TreeSHAP explanation | 6.0 ms | 7.0 ms | 6.2 ms |
| SQLite state write | 1.5 ms | 2.3 ms | 1.6 ms |
| Total scoring pipeline | 33.5 ms | 43.8 ms | 34.4 ms |
| Client-observed end-to-end HTTP | 53.4 ms | 67.2 ms | 53.0 ms |

The 50 requests completed in 2.66 seconds: 18.8 sequential events/second.

Prior to the indexed lookup optimization, behavioural feature generation
required scanning the full 428 MB reference Parquet on every request
(~248 ms median, ~96% of pipeline time). The SQLite B-tree indexed layer
reduced feature lookup to ~23 ms median (>10x improvement), with end-to-end
HTTP latency dropping from ~279 ms to ~53 ms and throughput increasing from
~3.6 to ~19 events/second. Feature computation semantics are unchanged and
validated by the `IndexedLookupRegressionTests` parity suite.

Environment: Windows 11, AMD64 (`AMD64 Family 25 Model 124 Stepping 0,
AuthenticAMD`), 12 logical CPUs, Python 3.14.2, NumPy 2.5.3, pandas 3.0.5,
XGBoost 3.4.1, FastAPI 0.141.1. This is a single local sample on one host;
the distributions are descriptive and not a capacity or latency guarantee.


## Container support

`Dockerfile` and `compose.yaml` provide a two-process local deployment with
an authenticated API, dashboard, read-only data mount, writable local model
and SQLite mount, loopback-only host ports, and an API readiness healthcheck.
The `.dockerignore` prevents large local datasets, model binaries, environment
secrets, and Git metadata from entering the build context.

With Docker Compose installed and `SYNDICAI_API_KEY` set in the shell:

```powershell
docker compose up --build
```

Open `http://127.0.0.1:8501`; the API schema is at
`http://127.0.0.1:8000/docs`. Stop with Ctrl+C. Data artifacts must already
exist locally under `data/processed/` and model artifacts under `models/`.
The compose API readiness endpoint verifies those artifacts are present.

### IBM Z / LinuxONE status

No IBM Z/LinuxONE host or container runtime was available for this work. The
local host tested here is Windows/AMD64. Docker and Podman are not installed,
so neither the Docker build nor an IBM deployment was run.

The native Python dependency stack is an additional concrete s390x blocker:
On 2026-10-03, the PyPI JSON release metadata checked for XGBoost 3.4.1,
NumPy 2.5.3, pandas 3.0.6, SciPy 1.18.1, scikit-learn 1.9.1, and PyArrow
25.0.1 listed no Linux/s390x wheels. The Anaconda API records checked also
contained no Linux/s390x builds for those packages, including their historical
records. Sources: [XGBoost](https://pypi.org/pypi/xgboost/json),
[NumPy](https://pypi.org/pypi/numpy/json),
[pandas](https://pypi.org/pypi/pandas/json),
[SciPy](https://pypi.org/pypi/scipy/json),
[scikit-learn](https://pypi.org/pypi/scikit-learn/json), and
[PyArrow](https://pypi.org/pypi/pyarrow/json) PyPI metadata; and the
[conda-forge XGBoost](https://api.anaconda.org/package/conda-forge/xgboost),
[NumPy](https://api.anaconda.org/package/conda-forge/numpy),
[pandas](https://api.anaconda.org/package/conda-forge/pandas),
[SciPy](https://api.anaconda.org/package/conda-forge/scipy),
[scikit-learn](https://api.anaconda.org/package/conda-forge/scikit-learn), and
[PyArrow](https://api.anaconda.org/package/conda-forge/pyarrow) records.
The generic Dockerfile must therefore not be represented as an IBM
Z-compatible image. No IBM hardware acceleration, IBM Z deployment, or
architecture-specific performance claim is made.

#### Concrete remaining adaptation path for IBM Z / LinuxONE

To validate and deploy SyndicAI onto IBM Z/LinuxONE in an enterprise setting, the following concrete adaptation steps are required:

1. **Hardware & Environment Access**:
   - Provision an actual s390x LPAR, LinuxONE virtual server, or Red Hat OpenShift on IBM Z cluster (or an automated QEMU `s390x` multi-arch CI runner for intermediate build checks).
2. **Native Toolchain & Wheel Resolution**:
   - Resolve native C/C++ and Fortran wheel availability for `s390x` (either via IBM Open Enterprise SDK for Python, conda-forge s390x channel where available, or custom multistage container builds with gcc/g++ and OpenBLAS).
3. **Inference Engine & Hardware Acceleration (NNPA / ONNX)**:
   - For low-latency inference on IBM z16/z15, evaluate exporting the fitted XGBoost Model B into ONNX format (`onnxmltools` / `skl2onnx`) or Treelite compiled C runtime.
   - Test `onnxruntime` built with IBM Integrated Accelerator for AI (NNPA) acceleration flags, decoupling scoring from heavy Python wheel constraints and maximizing hardware co-processor throughput.
4. **Endianness & Numerical Parity**:
   - Execute the strict parity test suite (`tests/test_online_scoring.py`) on `s390x` to verify that big-endian architecture conventions do not introduce precision drift or feature discrepancies relative to x86_64.
5. **Stateful Layer & End-to-End Verification**:
   - Verify SQLite index performance (`data/processed/reference_history.sqlite`) and run the full FastAPI live stream benchmark (`demo/benchmark_live.py`) on the target machine to measure real-world transaction throughput and latency.
