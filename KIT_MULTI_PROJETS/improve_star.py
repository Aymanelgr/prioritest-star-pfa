"""PRIORITEST-STAR — Phase 4 of ARCHITECTURE.md: the personal improvement of fusion / cross-view contrast (GUIDE §3).

All variants start from A5 (graph + similarity gate + cross-view InfoNCE, the best ladder step of Phase 1) and change
ONE thing. Same protocol as compare_fair.py (validation search, refit train+val, 10 seeds, single test pass).
  C1 label-aware InfoNCE   — same-label defective pairs removed from the negatives (false-negative fix)
  C2 SupCon hybrid         — C1 + other defective classes as soft positives across the views (src/model.py)
  G1 |zT − zS| gate        — the fusion gate also sees where the two views disagree
  G2 gate entropy          — regulariser keeping both views in use (α is ≈0.7 on the temporal view in STAR-R)
  C2G1                     — C2 + G1
Each is compared with A5 (re-run here with the same seeds) by paired class bootstrap and seed-paired Wilcoxon.

Usage: python improve_star.py --project ant
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
import run_star  # noqa: E402
from run_star import score_of, signed_log, releases_of, rank01  # noqa: E402
from compare_fair import tune_neural, evaluate, METRICS  # noqa: E402
from train_final import train_net  # noqa: E402
from train import load  # noqa: E402
import stats as S  # noqa: E402

A5 = run_star.VARIANTS['A5_plus_InfoNCE']
NEW = {'C0_A5_reference': {**A5},
       'C1_label_aware_InfoNCE': {**A5, 'label_aware': True},
       'C2_SupCon_hybrid': {**A5, 'contrast': 'supcon', 'sup_weight': .5},
       'G1_absdiff_gate': {**A5, 'gate_absdiff': True},
       'G2_gate_entropy': {**A5, 'gate_entropy': .05},
       'C2G1_SupCon_absdiff_gate': {**A5, 'contrast': 'supcon', 'sup_weight': .5, 'gate_absdiff': True}}
run_star.VARIANTS.update(NEW)       # tune_neural looks variants up by name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', default='ant'); ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--tune-seeds', type=int, nargs='+', default=[1, 2, 3]); ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--boot', type=int, default=2000); ap.add_argument('--out', default=str(ROOT / 'results' / 'phase4'))
    a = ap.parse_args(); torch.set_num_threads(2); t0 = time.time()
    tr, val, te = releases_of(a.project)
    data = {v: load(ROOT / 'data' / a.project / f'{a.project}-{v}.npz', 'cpu') for v in (tr, val)}
    for d in data.values():
        d['x'] = signed_log(d['x'])
    out = Path(a.out) / a.project; out.mkdir(parents=True, exist_ok=True)
    tuning = {n: tune_neural(n, data, tr, val, a.epochs, a.tune_seeds) for n in NEW}
    (out / 'tuning.json').write_text(json.dumps(tuning, indent=1, default=str))

    data[te] = load(ROOT / 'data' / a.project / f'{a.project}-{te}.npz', 'cpu'); data[te]['x'] = signed_log(data[te]['x'])
    dt = data[te]; y = dt['y'].numpy().astype(int); loc = dt['loc']; seeds = [100 + i for i in range(a.seeds)]
    per, ens, prob, alpha = {}, {}, {}, {}
    for n in NEW:
        t = tuning[n]; runs, al = [], []
        for s in seeds:
            net = train_net(t['cfg'], t['hp'], data, [tr, val], t['epochs'], s)
            with torch.no_grad():
                o = net(dt['x'], dt['edges'])
            runs.append(score_of(o, t['cfg']).numpy()); al.append(float(o['gate'][:, 0].mean()))
        per[n] = [evaluate(r, dt) for r in runs]; ens[n] = np.mean([rank01(r) for r in runs], 0); prob[n] = np.mean(runs, 0)
        alpha[n] = round(float(np.mean(al)), 3)
    rows, tests = {}, []
    for n in NEW:
        rows[n] = {m: [round(float(np.mean([x[m] for x in per[n]])), 4), round(float(np.std([x[m] for x in per[n]], ddof=1)), 4)]
                   for m in METRICS}
        rows[n]['mean_alpha_temporal'] = alpha[n]; rows[n]['val_AP'] = tuning[n]['val_AP']
        if n == 'C0_A5_reference':
            continue
        for m in ['AP', 'Popt_density']:
            src = prob if 'density' in m else ens
            d_, lo, hi, p = S.paired_bootstrap(src[n], src['C0_A5_reference'], y, loc, m, a.boot)
            pw = S.wilcoxon_seeds([x[m] for x in per[n]], [x[m] for x in per['C0_A5_reference']])
            tests.append({'model': n, 'metric': m, 'diff': round(d_, 4), 'CI95': [round(lo, 4), round(hi, 4)], 'p_bootstrap': round(p, 4),
                          'mean_seed_diff': round(float(np.mean([x[m] for x in per[n]]) - np.mean([x[m] for x in per['C0_A5_reference']])), 4),
                          'p_wilcoxon_seeds': round(pw, 4)})
    (out / 'results.json').write_text(json.dumps({'table': rows, 'tests_vs_A5': tests, 'seconds': round(time.time() - t0, 1)}, indent=1))
    np.savez_compressed(out / 'test_scores.npz', ids=dt['ids'], y=y, loc=loc, **ens, **{f'prob_{k}': v for k, v in prob.items()})
    print(f'\n=== {a.project} {te} (mean ± std over {a.seeds} seeds) ===')
    for n, r in rows.items():
        print(f"{n:28s} valAP={r['val_AP']:.4f} AP={r['AP'][0]:.4f}±{r['AP'][1]:.4f} Popt(dens)={r['Popt_density'][0]:.4f} "
              f"R@20%LOC(dens)={r['R@20LOC_density'][0]:.4f} ROC={r['ROC_AUC'][0]:.4f} α={r['mean_alpha_temporal']}")
    for t in tests:
        print(f"{t['model']:28s} {t['metric']:13s} Δens={t['diff']:+.4f} CI={t['CI95']} p={t['p_bootstrap']:.3f} | "
              f"Δseeds={t['mean_seed_diff']:+.4f} p_w={t['p_wilcoxon_seeds']:.3f}")
    print(f'total {time.time() - t0:.0f}s -> {out}')


if __name__ == '__main__':
    main()
