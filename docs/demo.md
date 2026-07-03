# Demo

The demo is a FastAPI service with a static browser frontend.

Install dependencies:

```bash
pip install -r requirements.txt
pip install -r demo/requirements.txt
```

Start the server:

```bash
MODEL_PATH=/path/to/FM9G4B-V python -m uvicorn demo.backend.app:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`.

The backend loads `MODEL_PATH` on startup. On a CPU-only machine this may be slow or fail for large models; GPU inference is recommended.
