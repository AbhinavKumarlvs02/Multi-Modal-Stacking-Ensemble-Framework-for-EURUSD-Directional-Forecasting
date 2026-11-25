"""Inference engine: EUR/USD T+5 Stacked Ensemble blind-test replay."""
from __future__ import annotations

import json
import math
import threading
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent.parent
MODELS, DATA = ROOT / "models", ROOT / "data"

REQUIRED = [MODELS / f for f in (
    "metadata.json", "thresholds.json", "scaler_struct.pkl", "lgbm_model.pkl",
    "catboost_model.cbm", "twotower_transformer.pth", "meta_logreg.pkl",
)] + [DATA / f for f in (
    "eurusd_features.csv", "daily_llm_aggregated.csv", "macro_fred.csv",
)]

OHLC = ["Open", "High", "Low", "Close"]
MACRO = ["US_10Y_Yield", "ECB_Deposit_Rate"]
TEST_FRAC = 0.15
STD_DDOF = 0  # 14d_std convention used when the GBMs were trained (numpy default)
BATCH = 512


# --------------------------------------------------------------------------- model
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 64, max_len: int = 50):
        super().__init__()
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TwoTowerFusion_Transformer(nn.Module):
    def __init__(self, struct_dim: int, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.struct_proj = nn.Linear(struct_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=True),
            num_layers,
        )
        self.sentiment_proj = nn.Sequential(nn.Linear(1, d_model // 2), nn.ReLU(), nn.Dropout(dropout))
        self.fusion_fc = nn.Sequential(
            nn.Linear(d_model + d_model // 2, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, x_combined: torch.Tensor) -> torch.Tensor:
        x_struct, x_unstruct = x_combined[:, :, :-1], x_combined[:, :, -1:]
        h = self.transformer(self.pos_encoder(self.struct_proj(x_struct)))[:, -1, :]
        s = self.sentiment_proj(x_unstruct[:, -1, :])
        return self.fusion_fc(torch.cat([h, s], dim=1))


# --------------------------------------------------------------------------- data
def _read(name: str) -> pd.DataFrame:
    df = pd.read_csv(DATA / name)
    dc = next((c for c in df.columns if c.lower() in {"date", "datetime", "timestamp"}), df.columns[0])
    idx = pd.to_datetime(df.pop(dc), utc=True).dt.tz_localize(None).dt.normalize()
    df.index = pd.DatetimeIndex(idx, name="date")
    return df[~df.index.duplicated(keep="last")].sort_index()


def _require(df: pd.DataFrame, cols: list[str], name: str) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"{name} is missing columns: {miss}")


def _prepare(struct_cols: list[str]) -> pd.DataFrame:
    df, sent, macro = _read("eurusd_features.csv"), _read("daily_llm_aggregated.csv"), _read("macro_fred.csv")
    _require(df, OHLC, "eurusd_features.csv")
    _require(sent, ["Sentiment"], "daily_llm_aggregated.csv")
    _require(macro, MACRO, "macro_fred.csv")

    df["Sentiment"] = sent["Sentiment"].reindex(df.index).fillna(0.0)  # no news -> neutral
    df[MACRO] = macro[MACRO].reindex(df.index, method="ffill")

    for c in OHLC:
        df[f"{c}_Return"] = df[c].pct_change(fill_method=None)
    df["Volatility_14d"] = df["Close_Return"].rolling(14).std()
    df["Momentum_5d"] = df["Close"].pct_change(5, fill_method=None)

    _require(df, struct_cols, "prepared features")
    return df.dropna(subset=[*struct_cols, "Sentiment", *OHLC])


def _p1(model, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)


def _thresholds() -> dict:
    try:
        t = json.loads((MODELS / "thresholds.json").read_text())["Stacked Ensemble"]
        return {"upper_90": float(t["upper_90"]), "lower_10": float(t["lower_10"])}
    except Exception:
        return {}


# --------------------------------------------------------------------------- inference
def _compute() -> dict:
    from catboost import CatBoostClassifier

    meta = json.loads((MODELS / "metadata.json").read_text())
    lb, cols = int(meta["lookback"]), list(meta["struct_cols"])
    thr = _thresholds()
    if not thr:
        raise ValueError("thresholds.json lacks ['Stacked Ensemble'] upper_90 / lower_10")

    df = _prepare(cols)
    n = len(df)
    start = int(n * (1 - TEST_FRAC))
    if start < lb - 1 or start >= n:
        raise ValueError(f"Dataset too short ({n} usable rows) for lookback={lb}")

    scaler = joblib.load(MODELS / "scaler_struct.pkl")
    arr = np.column_stack([scaler.transform(df[cols]), df["Sentiment"].to_numpy()])  # (n, D+1)
    win = sliding_window_view(arr, lb, axis=0).transpose(0, 2, 1)                  # (n-lb+1, lb, D+1)
    seq = win[np.arange(start, n) - lb + 1]                                        # (T, lb, D+1)
    tab = np.concatenate([seq[:, -1], seq.mean(1), seq.std(1, ddof=STD_DDOF)], axis=1)

    lgbm = joblib.load(MODELS / "lgbm_model.pkl")
    cb = CatBoostClassifier()
    cb.load_model(str(MODELS / "catboost_model.cbm"))
    meta_clf = joblib.load(MODELS / "meta_logreg.pkl")

    tf = TwoTowerFusion_Transformer(int(meta["struct_dim"]))
    state = torch.load(MODELS / "twotower_transformer.pth", map_location="cpu", weights_only=True)
    tf.load_state_dict(state.get("model_state_dict", state.get("state_dict", state)))
    tf.eval()

    x = np.ascontiguousarray(seq, dtype=np.float32)
    with torch.no_grad():
        tf_p = np.concatenate([
            tf(torch.from_numpy(x[i:i + BATCH])).squeeze(1).numpy() for i in range(0, len(x), BATCH)
        ])
    lgbm_p, cb_p = _p1(lgbm, tab), _p1(cb, tab)
    ens = meta_clf.predict_proba(np.column_stack([lgbm_p, cb_p, tf_p]))[:, 1]
    sig = np.where(ens >= thr["upper_90"], "LONG", np.where(ens <= thr["lower_10"], "SHORT", "HOLD"))

    d = df.iloc[start:]
    r5 = lambda v: round(float(v), 5)
    r6 = lambda v: round(float(v), 6)
    rows = [
        {"date": ts.strftime("%Y-%m-%d"), "open": r5(o), "high": r5(h), "low": r5(l), "close": r5(c),
         "signal": str(s), "ensemble_prob": r6(e), "lgbm_prob": r6(a), "cb_prob": r6(b), "tf_prob": r6(t)}
        for ts, o, h, l, c, s, e, a, b, t in zip(
            d.index, d["Open"], d["High"], d["Low"], d["Close"], sig, ens, lgbm_p, cb_p, tf_p)
    ]
    return {"thresholds": thr, "rows": rows}


_lock, _cache = threading.Lock(), {}


def _result() -> dict:
    key = tuple(p.stat().st_mtime_ns for p in REQUIRED)  # recompute if user swaps files
    with _lock:
        if _cache.get("key") != key:
            _cache.update(key=key, value=_compute())
        return _cache["value"]


# --------------------------------------------------------------------------- public API
def missing_files() -> list[str]:
    return [p.relative_to(ROOT).as_posix() for p in REQUIRED if not p.is_file()]


def status() -> dict:
    out = {"ready": False, "missing_files": missing_files(), "thresholds": _thresholds(),
           "total_days": 0, "error": None}
    if not out["missing_files"]:
        try:
            out.update(ready=True, total_days=len(_result()["rows"]))
        except Exception as e:  # surface config/shape problems instead of crashing
            out["error"] = f"{type(e).__name__}: {e}"
    return out


def replay() -> list[dict]:
    return _result()["rows"]
