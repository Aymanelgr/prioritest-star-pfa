"""PRIORITEST-STAR — step 5.5 of ARCHITECTURE.md: per-project tuning for projects without a validation release.

Camel and POI have only two target windows (train, test). Step 6.3 reused the settings frozen from Ant, and the graph
variants collapsed on POI. Here every model is tuned INSIDE the training release:
  * 3 stratified folds over the classes of the training release (seeded). The graph stays whole; the loss sees only the
    fold's training classes and AP is measured on the held-out classes (transductive node split, standard for GNNs).
  * Same grids as Phase 1 (lr × weight decay × Δ-input [+ λc, λr]); epoch = argmax of the fold-averaged smoothed AP curve.
  * Refit on ALL classes of the training release, 10 seeds, and ONE evaluation on the later test release.
The test release is loaded only after all choices are saved (R2). Tabular models get the same folds.
Usage: python retune_project.py --projects camel poi
"""
import argparse, itertools, json, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from run_star import VARIANTS, build_net, set_seed, signed_log, rank01, score_of  # noqa: E402
from model import cross_view_infonce, pairwise_rank_loss  # noqa: E402
from compare_fair import TABULAR, TAB_GRID, evaluate, METRICS  # noqa: E402
from train_final import tab_X  # noqa: E402
from train import load  # noqa: E402
import stats as S  # noqa: E402

SPLITS = {'camel': ('1.4', '1.6'), 'poi': ('2.5', '3.0')}
NEURAL = ['A2_TCN_BCE', 'A3_graph_classic_fusion_BCE', 'A4_similarity_fusion_BCE', 'A5_plus_InfoNCE', 'A6_STAR_full']


def masked_loss(net, cfg, hp, d, m):
    out = net(d['x'], d['edges']); y = d['y'][m]; pos = y.sum().clamp_min(1)
    loss = F.binary_cross_entropy_with_logits(out['logits'][m], y, pos_weight=((len(y) - pos) / pos).reshape(1))
    if cfg['lambda_r']:
        loss = loss + cfg['lambda_r'] * pairwise_rank_loss(out['rank'][m], y)
    if cfg['lambda_c']:
        loss = loss + cfg['lambda_c'] * cross_view_infonce(out['temporal'][m], out['structural'][m], hp['temperature'])
    return loss


def fit(cfg, hp, d, m, epochs, seed, val=None):
    set_seed(seed); net = build_net(cfg, 20, 'cpu')
    opt = torch.optim.Adam(net.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay']); curve = []
    yv = d['y'].numpy()[val] if val is not None else None
    for _ in range(epochs):
        net.train(); opt.zero_grad(); masked_loss(net, cfg, hp, d, m).backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
        if val is not None:
            net.eval()
            with torch.no_grad():
                s = score_of(net(d['x'], d['edges']), cfg).numpy()
            curve.append(S.metric_fn('AP')(s[val], yv, None))
    net.eval(); return net, np.array(curve)


def tune(name, d, folds, epochs):
    base = VARIANTS[name]; grid = {'lr': [1e-3, 3e-3], 'weight_decay': [1e-4, 1e-3], 'use_delta': [False, True]}
    if base['lambda_c']:
        grid['lambda_c'] = [.1, .25]
    if base['lambda_r']:
        grid['lambda_r'] = [.25, 1.]
    best = None
    for combo in itertools.product(*grid.values()):
        g = dict(zip(grid, combo))
        cfg = {**base, **{k: v for k, v in g.items() if k in ('use_delta', 'lambda_c', 'lambda_r')}}
        hp = {'lr': g['lr'], 'weight_decay': g['weight_decay'], 'temperature': .2}
        C = np.stack([fit(cfg, hp, d, torch.as_tensor(tr), epochs, 1, val=va)[1] for tr, va in folds])
        sm = np.convolve(C.mean(0), np.ones(5) / 5, mode='same'); sm[:10] = -1; ep = int(np.argmax(sm)) + 1
        if best is None or sm[ep - 1] > best['inner_AP']:
            best = {'cfg': cfg, 'hp': hp, 'epochs': ep, 'inner_AP': round(float(sm[ep - 1]), 4), 'grid_point': g}
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--projects', nargs='+', default=['camel', 'poi']); ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--epochs', type=int, default=150); ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--out', default=str(ROOT / 'results' / 'retune'))
    a = ap.parse_args(); torch.set_num_threads(2); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for P in a.projects:
        t0 = time.time(); tr_v, te_v = SPLITS[P]
        d = load(ROOT / 'data' / P / f'{P}-{tr_v}.npz', 'cpu'); d['x'] = signed_log(d['x']); y = d['y'].numpy().astype(int)
        folds = [(np.sort(i), np.sort(j)) for i, j in StratifiedKFold(3, shuffle=True, random_state=0).split(np.zeros(len(y)), y)]
        tuning = {}
        for n in NEURAL:
            tuning[n] = tune(n, d, folds, a.epochs)
            print(f'[{P}] {n:30s} inner AP={tuning[n]["inner_AP"]:.4f} ep={tuning[n]["epochs"]} {tuning[n]["grid_point"]}', flush=True)
        X = tab_X(d)
        for n in TABULAR:
            best = None
            for combo in itertools.product(*TAB_GRID[n].values()):
                p = dict(zip(TAB_GRID[n], combo))
                apv = np.mean([S.metric_fn('AP')(TABULAR[n](p, 0).fit(X[i], y[i]).predict_proba(X[j])[:, 1], y[j], None) for i, j in folds])
                if best is None or apv > best['inner_AP']:
                    best = {'params': p, 'inner_AP': round(float(apv), 4)}
            tuning[n] = best
            print(f'[{P}] {n:30s} inner AP={best["inner_AP"]:.4f} {best["params"]}', flush=True)
        (out / f'{P}_tuning.json').write_text(json.dumps(tuning, indent=1, default=str))

        # ---- refit on all training classes, single test pass ----
        dt = load(ROOT / 'data' / P / f'{P}-{te_v}.npz', 'cpu'); dt['x'] = signed_log(dt['x'])
        yt = dt['y'].numpy().astype(int); loc = dt['loc']; Xt = tab_X(dt); allm = torch.ones(len(y), dtype=torch.bool)
        seeds = [100 + i for i in range(a.seeds)]; runs = {}
        for n in NEURAL:
            t = tuning[n]; runs[n] = []
            for s in seeds:
                net, _ = fit(t['cfg'], t['hp'], d, allm, t['epochs'], s)
                with torch.no_grad():
                    runs[n].append(score_of(net(dt['x'], dt['edges']), t['cfg']).numpy())
        for n in TABULAR:
            runs[n] = [TABULAR[n](tuning[n]['params'], s).fit(X, y).predict_proba(Xt)[:, 1] for s in (seeds if n != 'B_LR' else seeds[:1])]
        ens = {k: np.mean([rank01(r) for r in v], 0) for k, v in runs.items()}; prob = {k: np.mean(v, 0) for k, v in runs.items()}
        ens['STAR_R_blend'] = np.mean([ens['A6_STAR_full'], rank01(ens['B_LR']), rank01(ens['B_ET'])], 0)
        prob['STAR_R_blend'] = np.mean([prob['A6_STAR_full'], prob['B_LR'], prob['B_ET']], 0)
        frozen = json.loads((ROOT / 'results' / 'transfer' / f'{P}.json').read_text())['table']
        table, tests = {}, []
        for k in ens:
            per = [evaluate(r, dt) for r in runs.get(k, [prob[k]])]
            row = {m: round(float(np.mean([p[m] for p in per])), 4) for m in METRICS}
            row['AP_std'] = round(float(np.std([p['AP'] for p in per], ddof=1)), 4) if len(per) > 1 else 0.
            e, lo, hi = S.bootstrap_ci(ens[k], yt, loc, 'AP', a.boot); row['AP_ens'] = [round(e, 4), round(lo, 4), round(hi, 4)]
            row['Popt_density_ens'] = round(float(S.metric_fn('Popt_density')(prob[k], yt, loc)), 4)
            row['frozen_from_ant_AP'] = frozen.get(k, {}).get('AP'); table[k] = row
            if k != 'B_LR':
                for m in ['AP', 'Popt_density']:
                    src = prob if 'density' in m else ens
                    d_, lo_, hi_, p = S.paired_bootstrap(src[k], src['B_LR'], yt, loc, m, a.boot)
                    tests.append({'model': k, 'vs': 'B_LR', 'metric': m, 'diff': round(d_, 4), 'CI95': [round(lo_, 4), round(hi_, 4)], 'p': round(p, 4)})
        for prev, cur in zip(['A1_HGB'] + NEURAL[:-1], NEURAL):
            d_, lo_, hi_, p = S.paired_bootstrap(ens[cur], ens[prev], yt, loc, 'AP', a.boot)
            tests.append({'model': cur, 'vs': prev, 'metric': 'AP', 'diff': round(d_, 4), 'CI95': [round(lo_, 4), round(hi_, 4)], 'p': round(p, 4)})
        res = {'project': P, 'train': tr_v, 'test': te_v, 'n_test': int(len(yt)), 'defective_test': int(yt.sum()),
               'protocol': '3-fold node split of the training release for tuning; refit on all; single test pass',
               'table': table, 'tests': tests, 'seconds': round(time.time() - t0, 1)}
        (out / f'{P}.json').write_text(json.dumps(res, indent=1))
        np.savez_compressed(out / f'{P}_test_scores.npz', ids=dt['ids'], y=yt, loc=loc, **ens, **{f'prob_{k}': v for k, v in prob.items()})
        print(f'\n=== {P}: train {tr_v} -> test {te_v} ({res["seconds"]}s) ===')
        for k, v in table.items():
            print(f"{k:28s} AP={v['AP']:.3f}±{v['AP_std']:.3f} ens {v['AP_ens'][0]:.3f} [{v['AP_ens'][1]:.3f},{v['AP_ens'][2]:.3f}] "
                  f"(frozen {v['frozen_from_ant_AP']})  ROC={v['ROC_AUC']:.3f} Popt(dens)={v['Popt_density_ens']:.3f}")
        for t in tests:
            print(f"  {t['model']:26s} vs {t['vs']:28s} {t['metric']:13s} Δ={t['diff']:+.4f} CI={t['CI95']} p={t['p']:.3f}")


if __name__ == '__main__':
    main()
