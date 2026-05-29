
import os
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F


MAT_PATH = "DH_FR1.mat"
MODEL_PATH = "model.pt"
EPS = 1e-6


def _torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _to_tensor(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def _prepare_stats(stats, device):
    out = {}
    for k, v in stats.items():
        out[k] = torch.as_tensor(v, dtype=torch.float32, device=device)
    return out


def _build_features(d, p_bs, stats):
    # d: (B, A), p_bs: (A, 2)
    B, A = d.shape

    d_mean = stats["d_mean"].view(1, A)
    d_std = stats["d_std"].view(1, A).clamp_min(EPS)
    log_mean = stats["log_d_mean"].view(1, A)
    log_std = stats["log_d_std"].view(1, A).clamp_min(EPS)

    d_norm = (d - d_mean) / d_std
    log_d = torch.log1p(torch.clamp(d, min=0.0))
    log_norm = (log_d - log_mean) / log_std

    bs_mean = stats["bs_mean"].view(1, 2)
    bs_std = stats["bs_std"].view(1, 2).clamp_min(EPS)
    bs_norm = (p_bs - bs_mean) / bs_std
    bs_norm = bs_norm.unsqueeze(0).expand(B, A, 2)

    d_feat = d_norm.unsqueeze(-1)
    log_feat = log_norm.unsqueeze(-1)
    cross = d_feat * bs_norm
    ones = torch.ones((B, A, 1), dtype=d.dtype, device=d.device)

    return torch.cat([d_feat, log_feat, bs_norm, cross, ones], dim=-1)


def _weighted_pairwise_wls(d, p_bs, weights=None, ridge=1e-3):
    # d: (B, A), p_bs: (A, 2), weights: (B, A)
    B, A_num = d.shape
    device = d.device
    dtype = d.dtype

    ii, jj = torch.triu_indices(A_num, A_num, offset=1, device=device)
    si = p_bs[ii]
    sj = p_bs[jj]

    A_mat = 2.0 * (sj - si)  # (P, 2)
    s_norm = torch.sum(p_bs * p_bs, dim=1)
    b = d[:, ii] ** 2 - d[:, jj] ** 2 - s_norm[ii].view(1, -1) + s_norm[jj].view(1, -1)

    if weights is None:
        pair_w = torch.ones_like(b)
    else:
        pair_w = weights[:, ii] * weights[:, jj]

    pair_w = torch.clamp(pair_w, min=1e-4, max=1e4)
    pair_w = pair_w / (pair_w.mean(dim=1, keepdim=True) + EPS)
    sqrt_w = torch.sqrt(pair_w)

    Aw = A_mat.unsqueeze(0) * sqrt_w.unsqueeze(-1)
    bw = b * sqrt_w

    H = torch.bmm(Aw.transpose(1, 2), Aw)
    H = H + ridge * torch.eye(2, dtype=dtype, device=device).unsqueeze(0)
    g = torch.bmm(Aw.transpose(1, 2), bw.unsqueeze(-1))

    sol = torch.linalg.solve(H, g).squeeze(-1)
    return sol


class AnchorReliabilityGNN(nn.Module):
    def __init__(self, num_anchors=18, feature_dim=7, hidden_dim=64, num_layers=3, max_delta=20.0):
        super().__init__()
        self.num_anchors = int(num_anchors)
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.max_delta = float(max_delta)

        self.log_scale = nn.Parameter(torch.zeros(self.num_anchors))
        self.bias = nn.Parameter(torch.zeros(self.num_anchors))

        self.node_in = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )

        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            for _ in range(self.num_layers)
        ])

        self.weight_head = nn.Linear(self.hidden_dim, 1)

        self.residual_head = nn.Sequential(
            nn.Linear(self.hidden_dim + 2, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 2),
        )

    def forward(self, d_raw, p_bs, stats):
        feat = _build_features(d_raw, p_bs, stats)
        h = self.node_in(feat)

        for layer in self.gnn_layers:
            h_global = h.mean(dim=1, keepdim=True).expand_as(h)
            h = h + layer(torch.cat([h, h_global], dim=-1))

        raw_w = self.weight_head(h).squeeze(-1)
        weights = F.softplus(raw_w) + 1e-3

        scale = torch.exp(torch.clamp(self.log_scale, min=-1.5, max=1.5)).view(1, -1)
        bias = self.bias.view(1, -1)
        d_cal = torch.clamp(d_raw * scale + bias, min=1e-3)

        p_wls = _weighted_pairwise_wls(d_cal, p_bs, weights)

        pos_mean = stats["pos_mean"].view(1, 2)
        pos_std = stats["pos_std"].view(1, 2).clamp_min(EPS)
        p_norm = (p_wls - pos_mean) / pos_std

        g = h.mean(dim=1)
        delta = torch.tanh(self.residual_head(torch.cat([g, p_norm], dim=-1))) * self.max_delta
        p_final = p_wls + delta

        return p_final, p_wls, weights, d_cal, delta


def _fallback_wls_numpy(d_hat, p_bs):
    # d_hat: (18, N), p_bs: (2, 18)
    device = torch.device("cpu")
    d = _to_tensor(d_hat.T, device)
    anchors = _to_tensor(p_bs.T, device)
    weights = 1.0 / torch.clamp(d, min=1.0) ** 2
    with torch.no_grad():
        p_hat = _weighted_pairwise_wls(d, anchors, weights=weights).cpu().numpy().T
    return p_hat.astype(float)


def predict_all(d_hat, p_bs, model_path=MODEL_PATH, batch_size=512):
    d_hat = np.asarray(d_hat, dtype=np.float32)
    p_bs = np.asarray(p_bs, dtype=np.float32)

    if d_hat.ndim != 2 or p_bs.ndim != 2:
        raise ValueError("d_hat and p_bs must be 2-D arrays.")
    if p_bs.shape[0] != 2:
        raise ValueError("p_bs must have shape (2, 18).")
    if d_hat.shape[0] != p_bs.shape[1]:
        raise ValueError("d_hat must have shape (num_anchor, num_user).")

    if not os.path.exists(model_path):
        return _fallback_wls_numpy(d_hat, p_bs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = _torch_load(model_path, device)

    cfg = ckpt.get("config", {})
    model = AnchorReliabilityGNN(
        num_anchors=int(cfg.get("num_anchors", d_hat.shape[0])),
        feature_dim=int(cfg.get("feature_dim", 7)),
        hidden_dim=int(cfg.get("hidden_dim", 64)),
        num_layers=int(cfg.get("num_layers", 3)),
        max_delta=float(cfg.get("max_delta", 20.0)),
    ).to(device)

    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()

    stats = _prepare_stats(ckpt["stats"], device)
    anchors = _to_tensor(p_bs.T, device)

    preds = []
    with torch.no_grad():
        for start in range(0, d_hat.shape[1], batch_size):
            end = min(start + batch_size, d_hat.shape[1])
            d_batch = _to_tensor(d_hat[:, start:end].T, device)
            p_pred, _, _, _, _ = model(d_batch, anchors, stats)
            preds.append(p_pred.detach().cpu().numpy())

    return np.concatenate(preds, axis=0).T.astype(float)


def your_algorithm(d_hat_u, p_bs):
    d_hat_u = np.asarray(d_hat_u, dtype=float).reshape(-1, 1)
    return predict_all(d_hat_u, p_bs)[:, 0]


def main():
    mat_path = "DH_FR1.mat"

    data = sio.loadmat(mat_path, squeeze_me=False)

    if "p_bs" in data:
        p_bs = np.asarray(data["p_bs"], dtype=float)
    elif "BS_positions" in data:
        p_bs = np.asarray(data["BS_positions"], dtype=float)
    else:
        keys = [k for k in data.keys() if not k.startswith("__")]

        raise KeyError(f"Cannot find BS position variable. Available keys: {keys}")


    d_hat = np.asarray(data["d_hat"], dtype=float)

    num_user = d_hat.shape[1]

    # GNN은 사용자별 loop보다 전체 batch 예측이 빠름
    p_hat = predict_all(d_hat, p_bs)

    if p_hat.shape != (2, num_user):
        raise ValueError(f"p_hat shape must be (2, {num_user}), got {p_hat.shape}")

    return p_hat


if __name__ == "__main__":
    p_hat = main()
    print("p_hat shape:", p_hat.shape)