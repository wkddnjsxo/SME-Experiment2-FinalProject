
import argparse
import os
import random
import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 1e-6


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_tensor(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def _prepare_stats(stats, device):
    out = {}
    for k, v in stats.items():
        out[k] = torch.as_tensor(v, dtype=torch.float32, device=device)
    return out


def _build_features(d, p_bs, stats):
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
    B, A_num = d.shape
    device = d.device
    dtype = d.dtype

    ii, jj = torch.triu_indices(A_num, A_num, offset=1, device=device)
    si = p_bs[ii]
    sj = p_bs[jj]

    A_mat = 2.0 * (sj - si)
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

    return torch.linalg.solve(H, g).squeeze(-1)


class AnchorReliabilityGNN(nn.Module):
    def __init__(self, num_anchors=18, feature_dim=7, hidden_dim=64, num_layers=3, max_delta=20.0, anchor_dropout=0.03):
        super().__init__()
        self.num_anchors = int(num_anchors)
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.max_delta = float(max_delta)
        self.anchor_dropout = float(anchor_dropout)

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

        if self.training and self.anchor_dropout > 0:
            keep = (torch.rand_like(weights) > self.anchor_dropout).float()
            weights = weights * keep + 1e-3

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


def load_mat(path):
    data = sio.loadmat(path, squeeze_me=False)

    if "p_bs" in data:
        p_bs = np.asarray(data["p_bs"], dtype=np.float32)
    elif "BS_positions" in data:
        p_bs = np.asarray(data["BS_positions"], dtype=np.float32)
    else:
        keys = [k for k in data.keys() if not k.startswith("__")]
        raise KeyError(f"Cannot find BS position variable. Available keys: {keys}")

    d_hat = np.asarray(data["d_hat"], dtype=np.float32)
    p = np.asarray(data["p"], dtype=np.float32)

    d = d_hat.T.copy()        # (N, 18)
    y = p.T.copy()            # (N, 2)
    anchors = p_bs.T.copy()   # (18, 2)

    return d, y, anchors


def make_stats(d_train, y_train, anchors, device):
    d_t = _to_tensor(d_train, device)
    y_t = _to_tensor(y_train, device)
    a_t = _to_tensor(anchors, device)

    log_d = torch.log1p(torch.clamp(d_t, min=0.0))

    stats = {
        "d_mean": d_t.mean(dim=0).detach().cpu(),
        "d_std": d_t.std(dim=0).clamp_min(EPS).detach().cpu(),
        "log_d_mean": log_d.mean(dim=0).detach().cpu(),
        "log_d_std": log_d.std(dim=0).clamp_min(EPS).detach().cpu(),
        "bs_mean": a_t.mean(dim=0).detach().cpu(),
        "bs_std": a_t.std(dim=0).clamp_min(EPS).detach().cpu(),
        "pos_mean": y_t.mean(dim=0).detach().cpu(),
        "pos_std": y_t.std(dim=0).clamp_min(EPS).detach().cpu(),
    }
    return stats


def batch_iter(indices, batch_size, shuffle=True):
    indices = np.asarray(indices)
    if shuffle:
        np.random.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield indices[start:start + batch_size]


@torch.no_grad()
def evaluate(model, d, y, anchors, stats, indices, batch_size, device):
    model.eval()
    errs = []
    losses = []

    for idx in batch_iter(indices.copy(), batch_size, shuffle=False):
        xb = _to_tensor(d[idx], device)
        yb = _to_tensor(y[idx], device)
        pred, p_wls, _, _, _ = model(xb, anchors, stats)

        err = torch.linalg.norm(pred - yb, dim=1)
        errs.append(err.detach().cpu())

        loss = F.smooth_l1_loss(pred, yb, beta=1.0)
        losses.append(loss.detach().cpu())

    err_all = torch.cat(errs).numpy()
    loss_val = torch.stack(losses).mean().item()
    return {
        "loss": loss_val,
        "mae": float(np.mean(err_all)),
        "median": float(np.median(err_all)),
        "rmse": float(np.sqrt(np.mean(err_all ** 2))),
        "p90": float(np.percentile(err_all, 90)),
    }


def train(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[INFO] device = {device}")

    d, y, anchors_np = load_mat(args.mat)
    num_samples, num_anchors = d.shape
    print(f"[INFO] samples={num_samples}, anchors={num_anchors}")

    perm = np.random.permutation(num_samples)
    val_size = max(1, int(num_samples * args.val_ratio))
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    stats_cpu = make_stats(d[train_idx], y[train_idx], anchors_np, device=torch.device("cpu"))
    stats = _prepare_stats(stats_cpu, device)
    anchors = _to_tensor(anchors_np, device)

    model = AnchorReliabilityGNN(
        num_anchors=num_anchors,
        feature_dim=7,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        max_delta=args.max_delta,
        anchor_dropout=args.anchor_dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_mae = float("inf")
    best_epoch = -1
    best_state = None
    wait = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for idx in batch_iter(train_idx.copy(), args.batch_size, shuffle=True):
            xb = _to_tensor(d[idx], device)
            yb = _to_tensor(y[idx], device)

            pred, p_wls, weights, d_cal, delta = model(xb, anchors, stats)

            loss_final = F.smooth_l1_loss(pred, yb, beta=1.0)
            loss_wls = F.smooth_l1_loss(p_wls, yb, beta=1.0)
            loss_delta = torch.mean(delta ** 2)
            loss_cal = torch.mean(model.log_scale ** 2) + 1e-4 * torch.mean(model.bias ** 2)

            loss = loss_final + args.lambda_wls * loss_wls + args.lambda_delta * loss_delta + args.lambda_cal * loss_cal

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_losses.append(loss.detach().cpu())

        if epoch == 1 or epoch % args.print_every == 0:
            val_metrics = evaluate(model, d, y, anchors, stats, val_idx, args.batch_size, device)
            train_loss = torch.stack(train_losses).mean().item()
            print(
                f"Epoch {epoch:04d} | train_loss={train_loss:.4f} | "
                f"val_mae={val_metrics['mae']:.4f} | val_median={val_metrics['median']:.4f}"
            )

            if val_metrics["mae"] < best_mae:
                best_mae = val_metrics["mae"]
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += args.print_every

            if wait >= args.patience:
                print(f"[INFO] Early stopping at epoch {epoch}. best_epoch={best_epoch}, best_val_mae={best_mae:.4f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    train_metrics = evaluate(model, d, y, anchors, stats, train_idx, args.batch_size, device)
    val_metrics = evaluate(model, d, y, anchors, stats, val_idx, args.batch_size, device)

    print("[RESULT] train:", train_metrics)
    print("[RESULT] val  :", val_metrics)

    checkpoint = {
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "stats": stats_cpu,
        "config": {
            "num_anchors": int(num_anchors),
            "feature_dim": 7,
            "hidden_dim": int(args.hidden_dim),
            "num_layers": int(args.num_layers),
            "max_delta": float(args.max_delta),
        },
        "metrics": {
            "best_epoch": int(best_epoch),
            "train": train_metrics,
            "val": val_metrics,
        },
    }

    torch.save(checkpoint, args.model)
    print(f"[INFO] saved model to {args.model}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mat", type=str, default="DH_FR1.mat")
    parser.add_argument("--model", type=str, default="model.pt")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--patience", type=int, default=80)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--max_delta", type=float, default=20.0)
    parser.add_argument("--anchor_dropout", type=float, default=0.03)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lambda_wls", type=float, default=0.3)
    parser.add_argument("--lambda_delta", type=float, default=0.002)
    parser.add_argument("--lambda_cal", type=float, default=0.001)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())