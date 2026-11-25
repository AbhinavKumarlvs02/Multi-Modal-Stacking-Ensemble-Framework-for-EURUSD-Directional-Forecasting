"""FastAPI app. Run from project root: uvicorn backend.main:app --port 8000"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

from backend import engine  # noqa: E402

FRONTEND = ROOT / "frontend"
FRONTEND.mkdir(exist_ok=True)

app = FastAPI(title="EUR/USD AI Trading Simulation")


@app.get("/api/status")
def api_status() -> dict:
    return engine.status()


@app.get("/api/replay-data")
def api_replay_data() -> list[dict]:
    st = engine.status()
    if not st["ready"]:
        raise HTTPException(status_code=503, detail=st)
    return engine.replay()


app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
