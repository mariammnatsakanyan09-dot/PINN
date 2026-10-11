"""Task 5 - neural-ODE / PINN for the Silverbox (Duffing) oscillator.

Physics:      y'' + a y' + b y + c y^3 = g u(t)           (eq. 1)
Model:        y' = v,   v' = alpha_theta(y, v, u)          (eq. 2)
              alpha_theta is a fully connected network that has to represent
              the whole restoring/damping/input force  g u - a v - b y - c y^3.

Pipeline
  1. excerpt [42650:46746] of SNLS80mV.mat, mean of u and of y removed
  2. first 3072 samples = training record, last 1024 = test record
  3. stage 0 (physics-informed warm start): match alpha_theta to the
     finite-difference acceleration at the training samples
  4. stage 1.. : multiple shooting.  Random windows of length L are
     integrated with RK4 (step dt = 1/fs, driven by the measured u) from the
     measured position and finite-difference velocity; loss = MSE of the
     simulated y against the measured y.  L grows from stage to stage.
  5. metric: ONE free run of the whole excerpt, started at sample 0 from
     (y[0], finite-difference velocity), driven only by u.  RMSE on the last
     1024 samples (test) and on the first 3072 samples (train).

Usage
  python pinn_silverbox.py run     --data SNLS80mV.mat --out runs [--only a,b] [--quick]
  python pinn_silverbox.py summary --out runs
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(1)

FS = 610.35
DT = 1.0 / FS
START = 42650
N_EXC = 4096
N_TRAIN = 3072
N_TEST = N_EXC - N_TRAIN


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load_signals(path):
    """Return V1 (input u) and V2 (output y) as 1-D float64 arrays."""
    import scipy.io as sio

    try:
        m = sio.loadmat(path)
        v1, v2 = m["V1"], m["V2"]
    except NotImplementedError:  # MATLAB v7.3 files are HDF5
        import h5py

        with h5py.File(path, "r") as f:
            v1, v2 = f["V1"][()], f["V2"][()]
    return (np.asarray(v1, dtype=np.float64).ravel(),
            np.asarray(v2, dtype=np.float64).ravel())


def prepare(path):
    v1, v2 = load_signals(path)
    if len(v1) != 131072 or len(v2) != 131072:
        print(f"WARNING: expected 131072 samples, got {len(v1)} / {len(v2)}")
    u = v1[START:START + N_EXC].copy()
    y = v2[START:START + N_EXC].copy()
    assert len(u) == N_EXC and len(y) == N_EXC
    u -= u.mean()
    y -= y.mean()
    return u, y


def fd_velocity(y):
    """Finite-difference velocity: central inside, 2nd-order one-sided at the ends."""
    v = np.empty_like(y)
    v[1:-1] = (y[2:] - y[:-2]) / (2 * DT)
    v[0] = (-3 * y[0] + 4 * y[1] - y[2]) / (2 * DT)
    v[-1] = (3 * y[-1] - 4 * y[-2] + y[-3]) / (2 * DT)
    return v



ACTS = {"tanh": nn.Tanh, "silu": nn.SiLU, "gelu": nn.GELU, "softplus": nn.Softplus}


class AccelNet(nn.Module):
    """alpha_theta(y, v, u): MLP on scaled inputs, output scaled back to V/s^2."""

    def __init__(self, hidden, act, sy, sv, su, sa):
        super().__init__()
        layers, d = [], 3
        for h in hidden:
            lin = nn.Linear(d, h)
            nn.init.xavier_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
            layers += [lin, ACTS[act]()]
            d = h
        out = nn.Linear(d, 1)
        nn.init.xavier_normal_(out.weight)
        out.weight.data *= 0.1
        nn.init.zeros_(out.bias)
        layers.append(out)
        self.net = nn.Sequential(*layers)
        for name, val in dict(sy=sy, sv=sv, su=su, sa=sa).items():
            self.register_buffer(name, torch.tensor(float(val)))

    def forward(self, y, v, u):
        x = torch.stack([y / self.sy, v / self.sv, u / self.su], dim=-1)
        return self.net(x).squeeze(-1) * self.sa


def rk4_step(model, y, v, u0, um, u1, dt=DT):
    """One classical RK4 step of  y' = v, v' = alpha(y, v, u)  with u known at
    t, t+dt/2 (linear interpolation) and t+dt."""
    k1y, k1v = v, model(y, v, u0)
    y2, v2 = y + 0.5 * dt * k1y, v + 0.5 * dt * k1v
    k2y, k2v = v2, model(y2, v2, um)
    y3, v3 = y + 0.5 * dt * k2y, v + 0.5 * dt * k2v
    k3y, k3v = v3, model(y3, v3, um)
    y4, v4 = y + dt * k3y, v + dt * k3v
    k4y, k4v = v4, model(y4, v4, u1)
    y_n = y + dt / 6.0 * (k1y + 2 * k2y + 2 * k3y + k4y)
    v_n = v + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return y_n, v_n


def rollout(model, y0, v0, useq):
    """y0, v0: (B,)   useq: (B, L+1) measured input  ->  simulated y: (B, L+1)."""
    L = useq.shape[1] - 1
    ys, y, v = [y0], y0, v0
    for k in range(L):
        u0, u1 = useq[:, k], useq[:, k + 1]
        y, v = rk4_step(model, y, v, u0, 0.5 * (u0 + u1), u1)
        ys.append(y)
    return torch.stack(ys, dim=1)


@torch.no_grad()
def free_run(model, y, u):
    """The metric's free run on the whole excerpt.  Reads y only at samples 0..2
    (position + finite-difference velocity at the first sample)."""
    model.eval()
    y0 = float(y[0])
    v0 = float((-3 * y[0] + 4 * y[1] - y[2]) / (2 * DT))
    yt = rollout(model,
                 torch.tensor([y0], dtype=torch.float32),
                 torch.tensor([v0], dtype=torch.float32),
                 torch.tensor(u, dtype=torch.float32)[None, :])
    model.train()
    return yt[0].numpy().astype(np.float64)


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def scores(model, y, u):
    yh = free_run(model, y, u)
    if not np.all(np.isfinite(yh)):
        return dict(train=float("inf"), test=float("inf")), yh
    return dict(train=rmse(yh[:N_TRAIN], y[:N_TRAIN]),
                test=rmse(yh[N_TRAIN:], y[N_TRAIN:])), yh



def make_opt(name, params, lr):
    if name == "adam":
        return torch.optim.Adam(params, lr=lr)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr)
    raise ValueError(name)


def run_config(name, cfg, u, y, outdir, seed=0, quick=False):
    t0 = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.makedirs(outdir, exist_ok=True)

    
    y_tr, u_tr = y[:N_TRAIN], u[:N_TRAIN]
    v_tr = fd_velocity(y_tr)
    a_tr = (v_tr[2:] - v_tr[:-2]) / (2 * DT)            
    sy, su, sv, sa = y_tr.std(), u_tr.std(), v_tr.std(), a_tr.std()

    model = AccelNet(cfg["hidden"], cfg["act"], sy, sv, su, sa)
    f32 = lambda a: torch.tensor(a, dtype=torch.float32)
    Y, U, V = f32(y_tr), f32(u_tr), f32(v_tr)
    log = []

    def say(msg):
        print(f"[{name}] {msg}", flush=True)

    
    steps0 = cfg.get("warm_steps", 1500)
    lr0 = cfg.get("warm_lr", 3e-3)
    if quick:
        steps0 = min(steps0, 200)
    if steps0 > 0:
        opt = make_opt(cfg["opt"], model.parameters(), lr0)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps0, eta_min=lr0 * 0.05)
        yk, vk, uk, ak = Y[1:-1], V[1:-1], U[1:-1], f32(a_tr)
        for it in range(steps0):
            loss = (((model(yk, vk, uk) - ak) / model.sa) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        say(f"warm start done, accel-matching loss {loss.item():.4e}")

    
    stages = cfg["stages"]
    for si, (L, steps, lr) in enumerate(stages):
        if quick:
            steps = max(30, steps // 15)
        B = max(8, min(128, 4096 // L))
        opt = make_opt(cfg["opt"], model.parameters(), lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr * 0.05)
        ar = torch.arange(L + 1)[None, :]
        t_stage = time.time()
        for it in range(steps):
            s = torch.randint(0, N_TRAIN - L, (B,))          
            idx = s[:, None] + ar
            ytar, useq = Y[idx], U[idx]
            yh = rollout(model, ytar[:, 0], V[s], useq)
            loss = (((yh[:, 1:] - ytar[:, 1:]) / model.sy) ** 2).mean()
            if not torch.isfinite(loss):
                say(f"stage {si} diverged at it {it}")
                break
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            if (it + 1) % max(1, steps // 4) == 0:
                say(f"stage {si} L={L} it {it + 1}/{steps} loss {loss.item():.4e} "
                    f"({time.time() - t_stage:.0f}s)")
        sc, _ = scores(model, y, u)
        log.append(dict(stage=si, L=L, steps=steps, lr=lr, **sc))
        say(f"after stage {si}: free-run RMSE train {sc['train']:.5f}  test {sc['test']:.5f}")

    sc, yh = scores(model, y, u)
    baseline = float(np.sqrt(np.mean(y[N_TRAIN:] ** 2)))
    res = dict(name=name, cfg=cfg, seed=seed, rmse_train=sc["train"], rmse_test=sc["test"],
               baseline_zero_test=baseline, n_params=sum(p.numel() for p in model.parameters()),
               seconds=time.time() - t0, stage_log=log, quick=quick)
    with open(os.path.join(outdir, "result.json"), "w") as f:
        json.dump(res, f, indent=1)
    np.savez(os.path.join(outdir, "pred.npz"), y=y, u=u, y_pred=yh)
    torch.save(model.state_dict(), os.path.join(outdir, "model.pt"))
    say(f"FINAL train {sc['train']:.5f}  test {sc['test']:.5f}  baseline {baseline:.5f}  "
        f"({res['seconds']:.0f}s)")
    return res



S_DEFAULT = [(16, 1500, 3e-3), (64, 1500, 1e-3), (256, 600, 3e-4)]
S_SHORT = [(16, 1500, 3e-3), (64, 1500, 1e-3)]
S_LONG = [(16, 1500, 3e-3), (64, 1500, 1e-3), (256, 800, 3e-4), (512, 400, 1e-4)]

CONFIGS = {
    "w32x2_tanh_adam":      dict(hidden=[32, 32],      act="tanh",     opt="adam",  stages=S_DEFAULT),
    "w64x2_tanh_adam":      dict(hidden=[64, 64],      act="tanh",     opt="adam",  stages=S_DEFAULT),
    "w64x3_tanh_adam":      dict(hidden=[64, 64, 64],  act="tanh",     opt="adam",  stages=S_DEFAULT),
    "w128x2_tanh_adam":     dict(hidden=[128, 128],    act="tanh",     opt="adam",  stages=S_DEFAULT),
    "w64x2_silu_adam":      dict(hidden=[64, 64],      act="silu",     opt="adam",  stages=S_DEFAULT),
    "w64x2_tanh_adamw":     dict(hidden=[64, 64],      act="tanh",     opt="adamw", stages=S_DEFAULT),
    "w64x2_tanh_adam_short": dict(hidden=[64, 64],     act="tanh",     opt="adam",  stages=S_SHORT),
    "w64x2_tanh_adam_long": dict(hidden=[64, 64],      act="tanh",     opt="adam",  stages=S_LONG),
}



def summary(outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = []
    for d in sorted(os.listdir(outdir)):
        p = os.path.join(outdir, d, "result.json")
        if os.path.isfile(p):
            with open(p) as f:
                runs.append(json.load(f))
    if not runs:
        sys.exit("no results found")
    runs.sort(key=lambda r: r["rmse_test"])
    best = runs[0]

    lines = ["| setting | hidden | act | optimizer | stages (L:steps:lr) | params | train RMSE | test RMSE |",
             "|---|---|---|---|---|---|---|---|"]
    for r in runs:
        c = r["cfg"]
        st = ", ".join(f"{L}:{n}:{lr:g}" for L, n, lr in c["stages"])
        lines.append(f"| {r['name']} | {c['hidden']} | {c['act']} | {c['opt']} | {st} | "
                     f"{r['n_params']} | {r['rmse_train']:.5f} | {r['rmse_test']:.5f} |")
    lines.append("")
    lines.append(f"Zero-prediction baseline (RMS of test output): {best['baseline_zero_test']:.5f} V")
    lines.append(f"Kept version: **{best['name']}**  -  free-run RMSE train {best['rmse_train']:.5f} V, "
                 f"test {best['rmse_test']:.5f} V")
    if any(r.get("quick") for r in runs):
        lines.append("")
        lines.append("NOTE: at least one run used --quick (shortened training).")
    with open(os.path.join(outdir, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    z = np.load(os.path.join(outdir, best["name"], "pred.npz"))
    t = np.arange(N_TRAIN, N_EXC) * DT
    ym, yp = z["y"][N_TRAIN:], z["y_pred"][N_TRAIN:]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True,
                                  gridspec_kw=dict(height_ratios=[3, 1]))
    ax.plot(t, ym, color="#2a6fb0", lw=1.4, label="measured y")
    ax.plot(t, yp, color="#d9822b", lw=1.4, ls="--", label="predicted y (free run)")
    ax.set_ylabel("output y  [V]")
    ax.set_title(f"Test segment: {best['name']}  (test RMSE {best['rmse_test']:.4f} V)")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.22 * (hi - lo))          # headroom so the legend never covers the signal
    ax.legend(frameon=False, loc="upper right", ncol=2)
    ax2.plot(t, yp - ym, color="#555555", lw=1.0)
    ax2.set_ylabel("error [V]")
    ax2.set_xlabel("time since start of excerpt  [s]")
    for a in (ax, ax2):
        a.spines[["top", "right"]].set_visible(False)
        a.grid(alpha=0.25, lw=0.6)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "best_test_plot.png"), dpi=160)
    print("wrote", os.path.join(outdir, "best_test_plot.png"))



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("run")
    pr.add_argument("--data", default="SNLS80mV.mat")
    pr.add_argument("--out", default="runs")
    pr.add_argument("--only", default="", help="comma-separated config names")
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--quick", action="store_true", help="tiny budgets, only to smoke-test")
    ps = sub.add_parser("summary")
    ps.add_argument("--out", default="runs")
    args = ap.parse_args()

    if args.cmd == "run":
        u, y = prepare(args.data)
        print(f"excerpt: {len(y)} samples, y RMS {y.std():.4f} V, "
              f"test-output RMS (zero baseline) {np.sqrt(np.mean(y[N_TRAIN:] ** 2)):.4f} V")
        names = [n for n in args.only.split(",") if n] or list(CONFIGS)
        for n in names:
            run_config(n, CONFIGS[n], u, y, os.path.join(args.out, n), seed=args.seed, quick=args.quick)
    else:
        summary(args.out)
