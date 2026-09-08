"""
Module 4c: LSTM sequence quantile features.
Trains a small LSTM on rolling daily sequences (default 90 days) to predict
weekly P10/P90 log-range. Outputs are stored as features
(lstm_p10/p90/spread), not as a replacement for the per-regime LightGBM
models.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from loguru import logger
from torch.utils.data import DataLoader, TensorDataset

from config import cfg

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

SEQ_LEN = 90
HIDDEN = 32
BATCH = 32
EPOCHS = 100
PATIENCE = 10
LR = 1e-3
SEED = 42


def build_sequences(seq_len: int | None = None) -> tuple:
    if seq_len is None:
        seq_len = SEQ_LEN
    daily = pd.read_parquet(DATA_DIR / "nifty_daily.parquet").sort_index()
    vix = pd.read_parquet(DATA_DIR / "india_vix_daily.parquet").sort_index()
    weekly = pd.read_parquet(DATA_DIR / "nifty_weekly.parquet").sort_index()
    vix_close = vix["close"] if "close" in vix.columns else vix.iloc[:, 0]
    feat = pd.DataFrame(index=daily.index)
    feat["log_ret"] = np.log(daily["close"] / daily["close"].shift(1))
    feat["log_hl"] = np.log(daily["high"] / daily["low"])
    feat["log_co"] = np.log(daily["close"] / daily["open"])
    feat["vix_level"] = vix_close.reindex(daily.index).ffill()
    feat["vix_change"] = feat["vix_level"].diff(1)
    feat = feat.dropna()
    target = np.log(weekly["high"] / weekly["low"]).rename("log_range")
    weeks, seqs, ys = [], [], []
    dates = feat.index
    for week_end in target.index:
        cutoff = week_end - pd.Timedelta(days=7)
        window = feat.loc[:cutoff].tail(seq_len)
        if len(window) < seq_len:
            continue
        mu, sd = window.mean(), window.std().clip(lower=1e-8)
        seqs.append(((window - mu) / sd).values.astype(np.float32))
        ys.append(float(target.loc[week_end]))
        weeks.append(week_end)
    X = np.stack(seqs)
    y = np.array(ys, dtype=np.float32)
    weeks = pd.DatetimeIndex(weeks, name="week_end")
    logger.info(f"LSTM sequences: {X.shape} weeks {weeks.min()} -> {weeks.max()}")
    return X, y, weeks


class QuantileLSTM(nn.Module):
    def __init__(self, n_feat: int, hidden: int = HIDDEN):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, 2)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(h[-1])


def pinball(pred: torch.Tensor, y: torch.Tensor, q: float) -> torch.Tensor:
    err = y - pred
    return torch.mean(torch.maximum(q * err, (q - 1) * err))


def train_lstm(seq_len: int | None = None, hidden: int | None = None, lr: float | None = None) -> dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if seq_len is None:
        seq_len = SEQ_LEN
    if hidden is None:
        hidden = HIDDEN
    if lr is None:
        lr = LR
    q10, q90 = cfg.alpha_mid_p10, cfg.alpha_mid_p90
    X, y, weeks = build_sequences(seq_len)
    n = len(X)
    split = int(n * 0.80)
    val_split = int(split * 0.90)
    Xtr, ytr = X[:val_split], y[:val_split]
    Xva, yva = X[val_split:split], y[val_split:split]
    Xte, yte = X[split:], y[split:]
    tr_loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)), batch_size=BATCH, shuffle=True)
    va_X, va_y = torch.from_numpy(Xva), torch.from_numpy(yva)
    model = QuantileLSTM(X.shape[2], hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state, bad = float("inf"), None, 0
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in tr_loader:
            opt.zero_grad()
            out = model(xb)
            loss = pinball(out[:, 0], yb, q10) + pinball(out[:, 1], yb, q90)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            out = model(va_X)
            vloss = float(pinball(out[:, 0], va_y, q10) + pinball(out[:, 1], va_y, q90))
        if vloss < best:
            best, best_state, bad = vloss, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                logger.info(f"Early stop at epoch {epoch} val={best:.6f}")
                break
    model.load_state_dict(best_state)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODELS_DIR / "lstm_seq.pt")
    model.eval()
    with torch.no_grad():
        full = model(torch.from_numpy(X)).numpy()
    p10, p90 = full[:, 0], full[:, 1]
    cross = p90 < p10
    if cross.sum() > 0:
        mid = (p10 + p90) / 2
        eps = np.maximum(0.002, 0.25 * np.maximum(p90 - p10, 0))
        p10 = np.where(cross, mid - eps, p10)
        p90 = np.where(cross, mid + eps, p90)
    df = pd.DataFrame({"lstm_p10": p10, "lstm_p90": p90, "lstm_spread": p90 - p10}, index=weeks)
    df.to_parquet(DATA_DIR / "lstm_quantiles.parquet")
    joblib.dump({"n_feat": X.shape[2], "seq_len": seq_len, "hidden": hidden, "q10": q10, "q90": q90}, MODELS_DIR / "lstm_seq_meta.pkl")

    def _report(yp10, yp90, yt, tag):
        cov = float(((yp10 <= yt) & (yt <= yp90)).mean())
        pb = float(np.mean(np.maximum(q10 * (yt - yp10), (q10 - 1) * (yt - yp10)) + np.maximum(q90 * (yt - yp90), (q90 - 1) * (yt - yp90))))
        logger.info(f"{tag}: coverage={cov:.2%} pinball={pb:.6f} n={len(yt)}")
        return cov, pb

    te_cov, te_pb = _report(p10[split:], p90[split:], y[split:], "test")
    tr_cov, tr_pb = _report(p10[:split], p90[:split], y[:split], "train")
    return {"n_weeks": n, "test_coverage": round(te_cov, 4), "test_pinball": round(te_pb, 6),
            "train_coverage": round(tr_cov, 4), "train_pinball": round(tr_pb, 6),
            "cross_fixed": int(cross.sum())}


if __name__ == "__main__":
    print(train_lstm())
