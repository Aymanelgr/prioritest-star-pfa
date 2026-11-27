"""PRIORITEST-STAR — Phase 2 of ARCHITECTURE.md: STAR-E, effort-aware ranking learned during training.

STAR (A5/A6) learns a risk score and becomes effort-aware only after training (score / LOC). STAR-E puts the
inspection cost inside the model and the loss:

    s_i = rank(z_i) - γ · log LOC_i                      (effort-aware score used directly for inspection order)

  E1  γ = 1 fixed, RankNet on (defective, clean) pairs            → learns log-risk-density
  E2  γ learned (init 1)                                          → the model chooses how much size to discount
  E3  E2 + "oracle-order" pairs: defective i vs defective j with LOC_i < LOC_j
                                                                  → matches the Popt oracle (defective first, small first)
The BCE detection head on z is kept (λ = 1), so the model still outputs a defect probability.

Same protocol as compare_fair.py: validation search on window 2 (test not loaded), epoch = argmax of the seed-averaged,
smoothed **validation Popt** curve, refit on windows 1+2 with 10 seeds, single test pass, class bootstrap.
Baselines for the comparison are read from results/phase1/<project>/test_scores.npz (inspected by score / LOC).

Usage: python effort_star.py --project ant
"""
import argparse, itertools, json, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from torch import nn
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from run_star import VARIANTS, build_net, set_seed, signed_log, releases_of, rank01, check_finite  # noqa: E402
from model import cross_view_infonce  # noqa: E402
from train import load  # noqa: E402
import stats as S  # noqa: E402

E_VARIANTS = {'E1_fixed_gamma': dict(learn_gamma=False, oracle_pairs=False),
              'E2_learned_gamma': dict(learn_gamma=True, oracle_pairs=False),
              'E3_learned_gamma_oracle_pairs': dict(learn_gamma=True, oracle_pairs=True)}


class STARE(nn.Module):
    def __init__(self, cfg, n_features, learn_gamma):
        super().__init__()
        self.net = build_net(cfg, n_features, 'cpu')
        self.log_gamma = nn.Parameter(torch.zeros(()), requires_grad=learn_gamma)   # γ = exp(log_gamma), init 1

    def forward(self, x, edges, logloc):
        out = self.net(x, edges)
        out['effort_score'] = out['rank'] - self.log_gamma.exp() * logloc
        return out


def pairs(y, loc, oracle, max_pairs=4000, gen=None):
    pos = torch.where(y > .5)[0]; neg = torch.where(y <= .5)[0]
    i = pos.repeat_interleave(len(neg)); j = neg.repeat(len(pos))
    if oracle and len(pos) > 1:   # defective vs defective: the smaller one must come first
        a = pos.repeat_interleave(len(pos)); b = pos.repeat(len(pos)); keep = loc[a] < loc[b]
        i = torch.cat([i, a[keep]]); j = torch.cat([j, b[keep]])
    if len(i) > max_pairs:
        take = torch.randperm(len(i), generator=gen)[:max_pairs]; i, j = i[take], j[take]
    return i, j


def loss_fn(m, d, ecfg, lc, lr_, temperature):
    out = m(d['x'], d['edges'], d['logloc']); check_finite(out, 'forward')
    y = d['y']; pos = y.sum().clamp_min(1)
    loss = F.binary_cross_entropy_with_logits(out['logits'], y, pos_weight=((len(y) - pos) / pos).reshape(1))
    i, j = pairs(y, d['locT'], ecfg['oracle_pairs'])
    if len(i):
        s = out['effort_score']; loss = loss + lr_ * F.softplus(-(s[i] - s[j])).mean()
    if lc:
        loss = loss + lc * cross_view_infonce(out['temporal'], out['structural'], temperature)
    return loss


def train(cfg, ecfg, hp, data, keys, epochs, seed, val=None):
    set_seed(seed); m = STARE(cfg, data[keys[0]]['x'].shape[-1], ecfg['learn_gamma'])
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=hp['lr'], weight_decay=hp['weight_decay'])
    curve = []
    for _ in range(epochs):
        m.train(); opt.zero_grad()
        loss = sum(loss_fn(m, data[k], ecfg, hp['lambda_c'], hp['lambda_r'], .2) for k in keys) / len(keys)
        loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
        if val is not None:
            curve.append(S.popt(predict(m, data[val])[0], data[val]['y'].numpy(), data[val]['loc']))
    m.eval(); return m, np.array(curve)


def predict(m, d):
    m.eval()
    with torch.no_grad():
        o = m(d['x'], d['edges'], d['logloc'])
    return o['effort_score'].numpy(), torch.sigmoid(o['logits']).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', default='ant'); ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--tune-seeds', type=int, nargs='+', default=[1, 2, 3]); ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--boot', type=int, default=2000); ap.add_argument('--out', default=str(ROOT / 'results' / 'phase2'))
    a = ap.parse_args(); torch.set_num_threads(2); t_all = time.time()
    tr, val, te = releases_of(a.project)

    def prep(v):
        d = load(ROOT / 'data' / a.project / f'{a.project}-{v}.npz', 'cpu'); d['x'] = signed_log(d['x'])
        d['locT'] = torch.as_tensor(np.maximum(d['loc'], 1), dtype=torch.float32); d['logloc'] = d['locT'].log()
        return d
    data = {v: prep(v) for v in (tr, val)}
    out = Path(a.out) / a.project; out.mkdir(parents=True, exist_ok=True)
    base = {**VARIANTS['A5_plus_InfoNCE'], 'lambda_r': .25}     # graph + similarity gate (+ InfoNCE when λc > 0)

    # 1. validation search (test not loaded): selection criterion = smoothed seed-averaged validation Popt
    grid = dict(lr=[1e-3, 3e-3], weight_decay=[1e-4, 1e-3], use_delta=[False, True], lambda_c=[0., .1], lambda_r=[.25, 1.])
    tuning = {}
    for name, ecfg in E_VARIANTS.items():
        best = None
        for combo in itertools.product(*grid.values()):
            g = dict(zip(grid, combo)); cfg = {**base, 'use_delta': g['use_delta'], 'lambda_c': g['lambda_c']}
            C = np.stack([train(cfg, ecfg, g, data, [tr], a.epochs, s, val)[1] for s in a.tune_seeds])
            mcurve = np.convolve(C.mean(0), np.ones(5) / 5, mode='same'); mcurve[:10] = -1
            ep = int(np.argmax(mcurve)) + 1
            if best is None or mcurve[ep - 1] > best['val_Popt']:
                best = {'cfg': cfg, 'hp': g, 'epochs': ep, 'val_Popt': round(float(mcurve[ep - 1]), 4)}
        tuning[name] = best
        print(f'[tune] {name:32s} val Popt={best["val_Popt"]:.4f} ep={best["epochs"]} {best["hp"]}', flush=True)
    (out / 'tuning.json').write_text(json.dumps(tuning, indent=1))

    # 2. refit on train+validation, 3. one test pass
    data[te] = prep(te); dt = data[te]; y = dt['y'].numpy().astype(int); loc = dt['loc']
    res, ens = {}, {}
    for name, ecfg in E_VARIANTS.items():
        t = tuning[name]; runs, probs, gammas = [], [], []
        for s in [100 + i for i in range(a.seeds)]:
            m, _ = train(t['cfg'], ecfg, t['hp'], data, [tr, val], t['epochs'], s)
            e, p = predict(m, dt); runs.append(e); probs.append(p); gammas.append(float(m.log_gamma.exp()))
        per = [{'Popt': S.popt(r, y, loc), 'R@20LOC': S.recall_at_loc(r, y, loc), 'IFA': S.ifa(r, y),
                'AP_detection_head': S.metric_fn('AP')(p, y, loc)} for r, p in zip(runs, probs)]
        ens[name] = np.mean([rank01(r) for r in runs], 0)
        res[name] = {k: [round(float(np.mean([x[k] for x in per])), 4), round(float(np.std([x[k] for x in per], ddof=1)), 4)]
                     for k in per[0]}
        res[name]['gamma'] = [round(float(np.mean(gammas)), 3), round(float(np.std(gammas, ddof=1)), 3)]
        res[name]['per_seed_Popt'] = [round(x['Popt'], 4) for x in per]

    # baselines from Phase 1, inspected by score / LOC (post-hoc density)
    p1 = np.load(ROOT / 'results' / 'phase1' / a.project / 'test_scores.npz')
    assert (p1['ids'] == dt['ids']).all()
    base_scores = {f'{k}+density': p1[f'prob_{k}'] / np.maximum(loc, 1) for k in ['B_RF', 'B_ET', 'B_LR', 'A5_plus_InfoNCE', 'A6_STAR_full']}
    table = {}
    for k, s in {**ens, **base_scores}.items():
        e, lo, hi = S.bootstrap_ci(s, y, loc, 'Popt', a.boot); r, rlo, rhi = S.bootstrap_ci(s, y, loc, 'R@20LOC', a.boot)
        table[k] = {'Popt': round(e, 4), 'Popt_CI': [round(lo, 4), round(hi, 4)],
                    'R@20LOC': round(r, 4), 'R@20LOC_CI': [round(rlo, 4), round(rhi, 4)], 'IFA': S.ifa(s, y)}
    tests = []
    for k in ens:
        for b in ['B_RF+density', 'B_LR+density', 'A5_plus_InfoNCE+density']:
            for m_ in ['Popt', 'R@20LOC']:
                d_, lo, hi, p = S.paired_bootstrap(ens[k], base_scores[b], y, loc, m_, a.boot)
                tests.append({'model': k, 'vs': b, 'metric': m_, 'diff': round(d_, 4), 'CI95': [round(lo, 4), round(hi, 4)], 'p': round(p, 4)})
    report = {'project': a.project, 'test': te, 'per_seed_mean_std': res, 'ensemble_table': table, 'tests': tests,
              'seconds': round(time.time() - t_all, 1)}
    (out / 'results.json').write_text(json.dumps(report, indent=1))
    np.savez_compressed(out / 'test_scores.npz', ids=dt['ids'], y=y, loc=loc, **ens)

    print(f'\n=== {a.project} {te}: STAR-E (mean ± std over {a.seeds} seeds) ===')
    for k, v in res.items():
        print(f"{k:32s} Popt={v['Popt'][0]:.3f}±{v['Popt'][1]:.3f} R@20%LOC={v['R@20LOC'][0]:.3f} IFA={v['IFA'][0]:.1f} "
              f"AP(det head)={v['AP_detection_head'][0]:.3f} γ={v['gamma'][0]:.2f}")
    print('\n=== ensembles vs post-hoc density baselines (class bootstrap 95% CI) ===')
    for k, v in table.items():
        print(f"{k:32s} Popt={v['Popt']:.3f} {v['Popt_CI']}  R@20%LOC={v['R@20LOC']:.3f} {v['R@20LOC_CI']}  IFA={v['IFA']}")
    print('\n=== paired tests ===')
    for t in tests:
        print(f"{t['model']:32s} vs {t['vs']:26s} {t['metric']:8s} Δ={t['diff']:+.4f} CI={t['CI95']} p={t['p']:.3f}")
    print(f'total {report["seconds"]}s -> {out}')


if __name__ == '__main__':
    main()
