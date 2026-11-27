"""PRIORITEST-STAR — Phase 1 of ARCHITECTURE.md: fair, tuned, significance-tested comparison.

Same procedure for EVERY model (no model gets a bigger search than another of its family):
  1. validation search: models trained on window 1 only, scored on window 2 (the test window is never loaded here);
     neural nets: grid lr × weight_decay × Δ-input (+ λc / λr when the variant uses them), 3 seeds,
     epoch = argmax of the seed-averaged, 5-epoch-smoothed validation-AP curve (≥ 10 epochs);
     tabular models: small grid of their main regularisation parameter;
  2. refit on windows 1+2 with the chosen settings, 10 seeds;
  3. ONE evaluation on window 3 (test).
Statistics: mean ± std over seeds, class-bootstrap 95 % CI of the 10-seed rank ensemble, Wilcoxon + Cliff's δ
between consecutive ablation steps, paired bootstrap of every model against logistic regression.

Usage: python compare_fair.py --project ant            (~10 min on 2 CPU cores)
"""
import argparse, csv, itertools, json, sys, time
from pathlib import Path
import numpy as np, torch
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from run_star import VARIANTS, score_of, signed_log, releases_of, rank01  # noqa: E402
from explore_val import curve  # noqa: E402
from train_final import train_net, tab_X  # noqa: E402
from train import load  # noqa: E402
import stats as S  # noqa: E402

NEURAL = ['A2_TCN_BCE', 'A3_graph_classic_fusion_BCE', 'A4_similarity_fusion_BCE', 'A5_plus_InfoNCE',
          'A6_STAR_full', 'A7_STAR_plus']
LADDER = ['A1_HGB'] + NEURAL[:5]                    # one mechanism per step (FICHE §6)
TABULAR = {
    'A1_HGB': lambda p, s: HistGradientBoostingClassifier(max_iter=p['it'], max_leaf_nodes=p['leaf'], learning_rate=.05,
                                                          class_weight='balanced', random_state=s),
    'B_LR': lambda p, s: LogisticRegression(C=p['C'], class_weight='balanced', max_iter=5000),
    'B_RF': lambda p, s: RandomForestClassifier(500, min_samples_leaf=p['leaf'], class_weight='balanced_subsample',
                                                random_state=s, n_jobs=2),
    'B_ET': lambda p, s: ExtraTreesClassifier(500, min_samples_leaf=p['leaf'], class_weight='balanced_subsample',
                                              random_state=s, n_jobs=2),
}
TAB_GRID = {'A1_HGB': {'it': [50, 100, 200], 'leaf': [4, 7, 15]}, 'B_LR': {'C': [.003, .01, .03, .1, .3, 1.]},
            'B_RF': {'leaf': [1, 3, 5, 10]}, 'B_ET': {'leaf': [1, 3, 5, 10]}}
KNN_N = [5, 10, 15, 20, 30, 50]
METRICS = ['AP', 'ROC_AUC', 'Popt', 'Popt_density', 'R@20LOC_density', 'IFA']


def knn_cosine(Xtr, ytr, Xq, N):
    """Rhazi-style decision rule on metrics: cosine similarity to training classes, Top-N, weighted vote.
    Score = Σ sim(defective neighbours) / Σ |sim(all neighbours)| — continuous, so it can be ranked."""
    a = Xtr / np.linalg.norm(Xtr, axis=1, keepdims=True).clip(1e-9)
    b = Xq / np.linalg.norm(Xq, axis=1, keepdims=True).clip(1e-9)
    sim = b @ a.T; idx = np.argsort(-sim, axis=1)[:, :N]; s = np.take_along_axis(sim, idx, 1)
    return (s * ytr[idx]).sum(1) / np.abs(s).sum(1).clip(1e-9)


def evaluate(scores, d):
    y = d['y'].numpy().astype(int); loc = d['loc']
    return {m: float(S.metric_fn(m)(scores, y, loc)) for m in METRICS}


def smoothed_epoch(C):
    m = np.convolve(C.mean(0), np.ones(5) / 5, mode='same'); m[:10] = -1
    ep = int(np.argmax(m)) + 1
    return ep, float(m[ep - 1])


def tune_neural(name, data, tr, val, epochs, seeds):
    base = VARIANTS[name]
    grid = {'lr': [1e-3, 3e-3], 'weight_decay': [1e-4, 1e-3], 'use_delta': [False, True]}
    if base['lambda_c']:
        grid['lambda_c'] = [.1, .25]
    if base['lambda_r']:
        grid['lambda_r'] = [.25, 1.]
    best = None
    for combo in itertools.product(*grid.values()):
        g = dict(zip(grid, combo))
        cfg = {**base, **{k: v for k, v in g.items() if k in ('use_delta', 'lambda_c', 'lambda_r')}}
        hp = {'lr': g['lr'], 'weight_decay': g['weight_decay'], 'temperature': .2}
        C = np.stack([curve(cfg, hp, data, tr, val, epochs, s, 'cpu', True) for s in seeds])
        ep, ap = smoothed_epoch(C)
        if best is None or ap > best['val_AP']:
            best = {'cfg': cfg, 'hp': hp, 'epochs': ep, 'val_AP': round(ap, 4), 'grid_point': g}
    print(f'  [tune] {name:30s} val AP={best["val_AP"]:.4f} ep={best["epochs"]} {best["grid_point"]}', flush=True)
    return best


def tune_tabular(name, Xtr, ytr, Xv, yv):
    best = None
    for combo in itertools.product(*TAB_GRID[name].values()):
        p = dict(zip(TAB_GRID[name], combo))
        ap = S.metric_fn('AP')(TABULAR[name](p, 0).fit(Xtr, ytr).predict_proba(Xv)[:, 1], yv, None)
        if best is None or ap > best['val_AP']:
            best = {'params': p, 'val_AP': round(float(ap), 4)}
    print(f'  [tune] {name:30s} val AP={best["val_AP"]:.4f} {best["params"]}', flush=True)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', default='ant'); ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--tune-seeds', type=int, nargs='+', default=[1, 2, 3]); ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--boot', type=int, default=2000); ap.add_argument('--out', default=str(ROOT / 'results' / 'phase1'))
    a = ap.parse_args(); torch.set_num_threads(2); t_all = time.time()
    tr, val, te = releases_of(a.project)
    data = {v: load(ROOT / 'data' / a.project / f'{a.project}-{v}.npz', 'cpu') for v in (tr, val)}
    for d in data.values():
        d['x'] = signed_log(d['x'])
    out = Path(a.out) / a.project; out.mkdir(parents=True, exist_ok=True)
    seeds = [100 + i for i in range(a.seeds)]

    # ---------------- 1. validation search (test window NOT loaded yet) ----------------
    print(f'== validation search: train {tr} -> validation {val}')
    Xtr, ytr = tab_X(data[tr]), data[tr]['y'].numpy(); Xv, yv = tab_X(data[val]), data[val]['y'].numpy()
    tuning = {n: tune_neural(n, data, tr, val, a.epochs, a.tune_seeds) for n in NEURAL}
    tuning.update({n: tune_tabular(n, Xtr, ytr, Xv, yv) for n in TABULAR})
    knn = max(KNN_N, key=lambda N: S.metric_fn('AP')(knn_cosine(Xtr, ytr, Xv, N), yv, None))
    tuning['B_kNN_cosine'] = {'params': {'N': knn}}
    print(f'  [tune] B_kNN_cosine                   N={knn}')
    (out / 'tuning.json').write_text(json.dumps(tuning, indent=1, default=str))

    # ---------------- 2. refit on train+validation, 3. single test evaluation ----------------
    data[te] = load(ROOT / 'data' / a.project / f'{a.project}-{te}.npz', 'cpu'); data[te]['x'] = signed_log(data[te]['x'])
    dt = data[te]; Xte = tab_X(dt); keys = [tr, val]
    Xall = np.concatenate([Xtr, Xv]); yall = np.concatenate([ytr, yv])
    per_seed, ens = {}, {}
    for n in NEURAL:
        t = tuning[n]; t0 = time.time(); runs = []
        for s in seeds:
            net = train_net(t['cfg'], t['hp'], data, keys, t['epochs'], s)
            with torch.no_grad():
                runs.append(score_of(net(dt['x'], dt['edges']), t['cfg']).numpy())
        per_seed[n] = [evaluate(r, dt) for r in runs]; ens[n] = np.mean([rank01(r) for r in runs], 0)
        print(f'  [test] {n:30s} AP={np.mean([r["AP"] for r in per_seed[n]]):.4f} ({time.time() - t0:.0f}s)', flush=True)
    for n in TABULAR:
        runs = [TABULAR[n](tuning[n]['params'], s).fit(Xall, yall).predict_proba(Xte)[:, 1] for s in seeds]
        per_seed[n] = [evaluate(r, dt) for r in runs]; ens[n] = np.mean([rank01(r) for r in runs], 0)
    r = knn_cosine(Xall, yall, Xte, knn); per_seed['B_kNN_cosine'] = [evaluate(r, dt)] * len(seeds); ens['B_kNN_cosine'] = r
    # STAR-R blend as in train_final.py: A6-tuned STAR ensemble + LR + ET, rank average (fixed a priori here)
    ens['STAR_R_blend'] = np.mean([ens['A6_STAR_full'], rank01(ens['B_LR']), rank01(ens['B_ET'])], 0)
    per_seed['STAR_R_blend'] = [evaluate(ens['STAR_R_blend'], dt)] * len(seeds)   # deterministic given ensembles
    np.savez_compressed(out / 'test_scores.npz', ids=dt['ids'], y=dt['y'].numpy(), loc=dt['loc'],
                        **{k: v for k, v in ens.items()})

    # ---------------- statistics ----------------
    y = dt['y'].numpy().astype(int); loc = dt['loc']; rows = []
    for n in list(per_seed):
        row = {'model': n, 'n_seeds': len(seeds), 'val_AP': tuning.get(n, {}).get('val_AP')}
        for m in METRICS:
            v = [x[m] for x in per_seed[n]]
            row[f'{m}_mean'] = round(float(np.mean(v)), 4); row[f'{m}_std'] = round(float(np.std(v, ddof=1)), 4)
        for m in ['AP', 'Popt_density', 'R@20LOC_density']:
            e, lo, hi = S.bootstrap_ci(ens[n], y, loc, m, a.boot)
            row[f'{m}_ens'] = round(e, 4); row[f'{m}_ens_CI'] = f'[{lo:.3f}, {hi:.3f}]'
        rows.append(row)
    with (out / 'table.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    tests = {'ladder': [], 'vs_LR': []}
    for prev, cur in zip(LADDER, LADDER[1:]):
        for m in ['AP', 'Popt_density']:
            a_, b_ = [x[m] for x in per_seed[cur]], [x[m] for x in per_seed[prev]]
            dlt, mag = S.cliffs_delta(a_, b_)
            tests['ladder'].append({'step': f'{prev} -> {cur}', 'metric': m, 'mean_diff': round(float(np.mean(a_) - np.mean(b_)), 4),
                                    'wilcoxon_p': round(S.wilcoxon_seeds(a_, b_), 4), 'cliffs_delta': round(dlt, 3), 'magnitude': mag})
    for n in ens:
        if n == 'B_LR':
            continue
        for m in ['AP', 'Popt_density']:
            d_, lo, hi, p = S.paired_bootstrap(ens[n], ens['B_LR'], y, loc, m, a.boot)
            tests['vs_LR'].append({'model': n, 'metric': m, 'diff': round(d_, 4), 'CI95': [round(lo, 4), round(hi, 4)], 'p': round(p, 4)})
    tests['seconds'] = round(time.time() - t_all, 1)
    (out / 'tests.json').write_text(json.dumps(tests, indent=1))

    print(f'\n=== {a.project} test {te}: mean ± std over {len(seeds)} seeds | 10-seed ensemble AP [95% CI] ===')
    for r_ in rows:
        print(f"{r_['model']:30s} AP={r_['AP_mean']:.3f}±{r_['AP_std']:.3f}  ens {r_['AP_ens']:.3f} {r_['AP_ens_CI']}  "
              f"Popt(dens)={r_['Popt_density_mean']:.3f}  R@20%LOC(dens)={r_['R@20LOC_density_mean']:.3f}  "
              f"Popt(risk)={r_['Popt_mean']:.3f}  IFA={r_['IFA_mean']:.1f}")
    print('\n-- ablation ladder (Wilcoxon over seeds) --')
    for t in tests['ladder']:
        print(f"{t['step']:58s} {t['metric']:13s} Δ={t['mean_diff']:+.4f} p={t['wilcoxon_p']:.3f} δ={t['cliffs_delta']:+.2f} ({t['magnitude']})")
    print('\n-- vs logistic regression (paired class bootstrap) --')
    for t in tests['vs_LR']:
        print(f"{t['model']:30s} {t['metric']:13s} Δ={t['diff']:+.4f} CI={t['CI95']} p={t['p']:.3f}")
    print(f'total {tests["seconds"]}s -> {out}')


if __name__ == '__main__':
    main()
