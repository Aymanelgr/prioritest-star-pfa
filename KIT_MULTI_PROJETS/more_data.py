"""PRIORITEST-STAR — Phase 3 of ARCHITECTURE.md: does more (cross-project) data unlock STAR?

Ant has 118 training classes / 20 defective. The other 9 PROMISE projects of the kit hold ~3 600 labelled classes
(every prepared window of camel, jedit, log4j, lucene, poi, synapse, velocity, xalan, xerces — NEVER Ant).
Each project is already standardised on its own first window (project-wise normalisation), then signed-log.

Settings compared, all evaluated once on Ant 1.7 after selection on Ant 1.6:
  X0  Ant only                      — Phase 1 reference (train 1.5; refit 1.5+1.6)
  X1  Others only (zero-shot)       — trained on the other projects only (cross-project prediction, FICHE bonus)
  X2  Others + Ant, joint           — every training window is one cohort; Ant cohorts get weight --ant-weight
  X3  Others pre-train → Ant fine-tune
Models: STAR A5 (graph + similarity gate + InfoNCE, Phase-1 tuned hyper-parameters) and the tabular LR / RF.
Neural epochs (and fine-tuning epochs) are chosen on the seed-averaged smoothed Ant-1.6 AP curve.

Usage: python more_data.py --project ant
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from run_star import build_net, cohort_loss, score_of, set_seed, signed_log, rank01  # noqa: E402
from train_final import tab_X  # noqa: E402
from train import load  # noqa: E402
import stats as S  # noqa: E402

METRICS = ['AP', 'ROC_AUC', 'Popt_density', 'R@20LOC_density']


def windows(exclude):
    m = json.loads((ROOT / 'data' / 'manifest_multi.json').read_text(encoding='utf-8'))
    out = {}
    for p, info in m['projects'].items():
        if p == exclude:
            continue
        for t in info['targets']:
            if t['status'] == 'prepared':
                d = load(ROOT / 'data' / p / f'{p}-{t["version"]}.npz', 'cpu'); d['x'] = signed_log(d['x'])
                out[f'{p}-{t["version"]}'] = d
    return out


def fit(cfg, hp, data, keys, weights, epochs, seed, init=None, val=None, lr_scale=1.):
    set_seed(seed); net = build_net(cfg, 20, 'cpu')
    if init is not None:
        net.load_state_dict(init)
    opt = torch.optim.Adam(net.parameters(), lr=hp['lr'] * lr_scale, weight_decay=hp['weight_decay'])
    w = torch.tensor([weights[k] for k in keys]); w = w / w.sum(); curve = []
    for _ in range(epochs):
        net.train(); opt.zero_grad()
        loss = sum(wi * cohort_loss(net, cfg, hp, data[k], True) for wi, k in zip(w, keys))
        loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
        if val is not None:
            curve.append(S.metric_fn('AP')(scores(net, cfg, val), val['y'].numpy(), None))
    net.eval(); return net, np.array(curve)


def scores(net, cfg, d):
    net.eval()
    with torch.no_grad():
        return score_of(net(d['x'], d['edges']), cfg).numpy()


def pick_epoch(curves):
    m = np.convolve(np.mean(curves, 0), np.ones(5) / 5, mode='same'); m[:10] = -1
    e = int(np.argmax(m)) + 1
    return e, round(float(m[e - 1]), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', default='ant'); ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--tune-seeds', type=int, nargs='+', default=[1, 2, 3]); ap.add_argument('--epochs', type=int, default=150)
    ap.add_argument('--ant-weight', type=float, default=.5, help='total loss weight of target-project cohorts in X2')
    ap.add_argument('--boot', type=int, default=2000); ap.add_argument('--out', default=str(ROOT / 'results' / 'phase3'))
    a = ap.parse_args(); torch.set_num_threads(2); t0 = time.time()
    P = a.project
    tr, val, te = [t['version'] for t in json.loads((ROOT / 'data' / 'manifest_multi.json').read_text())['projects'][P]['targets']]
    data = {f'{P}-{v}': load(ROOT / 'data' / P / f'{P}-{v}.npz', 'cpu') for v in (tr, val)}
    for d in data.values():
        d['x'] = signed_log(d['x'])
    others = windows(P); data.update(others); O = list(others)
    n_other = sum(len(d['y']) for d in others.values()); p_other = int(sum(d['y'].sum() for d in others.values()))
    print(f'other projects: {len(O)} windows, {n_other} classes, {p_other} defective; target train {P}-{tr}: {len(data[f"{P}-{tr}"]["y"])}')
    T, V = f'{P}-{tr}', f'{P}-{val}'
    t1 = json.loads((ROOT / 'results' / 'phase1' / P / 'tuning.json').read_text())['A5_plus_InfoNCE']
    cfg, hp = t1['cfg'], t1['hp']
    out = Path(a.out) / P; out.mkdir(parents=True, exist_ok=True)

    def weights(keys):   # target cohorts share ant_weight, other cohorts share 1 - ant_weight (size-proportional)
        tk = [k for k in keys if k.startswith(P + '-')]; ok = [k for k in keys if k not in tk]
        w = {k: a.ant_weight / len(tk) for k in tk} if tk else {}
        tot = sum(len(data[k]['y']) for k in ok)
        w.update({k: (1 - a.ant_weight if tk else 1.) * len(data[k]['y']) / tot for k in ok}); return w

    # ---------- validation selection (Ant test not loaded) ----------
    sel = {}
    sel['X0_ant_only'] = {'epochs': t1['epochs'], 'val_AP': t1['val_AP']}
    for name, keys in [('X1_others_only', O), ('X2_joint', O + [T])]:
        C = [fit(cfg, hp, data, keys, weights(keys), a.epochs, s, val=data[V])[1] for s in a.tune_seeds]
        sel[name] = dict(zip(['epochs', 'val_AP'], pick_epoch(C)))
        print(f'[select] {name:16s} ep={sel[name]["epochs"]} val AP={sel[name]["val_AP"]}', flush=True)
    pre_ep = sel['X1_others_only']['epochs']
    pre = {s: fit(cfg, hp, data, O, weights(O), pre_ep, s)[0].state_dict() for s in a.tune_seeds}
    C = [fit(cfg, hp, data, [T], {T: 1.}, a.epochs, s, init=pre[s], val=data[V], lr_scale=.3)[1] for s in a.tune_seeds]
    sel['X3_pretrain_finetune'] = dict(zip(['ft_epochs', 'val_AP'], pick_epoch(C))); sel['X3_pretrain_finetune']['pre_epochs'] = pre_ep
    print(f'[select] X3 pretrain {pre_ep} ep + finetune {sel["X3_pretrain_finetune"]["ft_epochs"]} ep, val AP={sel["X3_pretrain_finetune"]["val_AP"]}', flush=True)
    # tabular: pooled vs target-only, validation AP
    Xt, yt = tab_X(data[T]), data[T]['y'].numpy(); Xv, yv = tab_X(data[V]), data[V]['y'].numpy()
    Xo = np.concatenate([tab_X(others[k]) for k in O]); yo = np.concatenate([others[k]['y'].numpy() for k in O])
    wv = np.r_[np.full(len(yo), (1 - a.ant_weight) / len(yo)), np.full(len(yt), a.ant_weight / len(yt))]; wv *= len(wv) / wv.sum()
    for nm, (X, y, W) in {'tab_ant_only': (Xt, yt, None), 'tab_others_only': (Xo, yo, None), 'tab_joint': (np.r_[Xo, Xt], np.r_[yo, yt], wv)}.items():
        lr = LogisticRegression(C=.03, class_weight='balanced', max_iter=5000).fit(X, y, sample_weight=W)
        sel[nm] = {'LR_val_AP': round(float(S.metric_fn('AP')(lr.predict_proba(Xv)[:, 1], yv, None)), 4)}
    (out / 'selection.json').write_text(json.dumps(sel, indent=1))

    # ---------- refit (target train + validation where applicable) and single test pass ----------
    dt = load(ROOT / 'data' / P / f'{P}-{te}.npz', 'cpu'); dt['x'] = signed_log(dt['x'])
    y = dt['y'].numpy().astype(int); loc = dt['loc']; seeds = [100 + i for i in range(a.seeds)]
    TV = [T, V]; runs = {}
    runs['STAR_X0_ant_only'] = [scores(fit(cfg, hp, data, TV, {T: 1, V: 1}, sel['X0_ant_only']['epochs'], s)[0], cfg, dt) for s in seeds]
    runs['STAR_X1_others_only'] = [scores(fit(cfg, hp, data, O, weights(O), sel['X1_others_only']['epochs'], s)[0], cfg, dt) for s in seeds]
    runs['STAR_X2_joint'] = [scores(fit(cfg, hp, data, O + TV, weights(O + TV), sel['X2_joint']['epochs'], s)[0], cfg, dt) for s in seeds]
    x3 = sel['X3_pretrain_finetune']; runs['STAR_X3_pretrain_finetune'] = []
    for s in seeds:
        init = fit(cfg, hp, data, O, weights(O), x3['pre_epochs'], s)[0].state_dict()
        runs['STAR_X3_pretrain_finetune'].append(scores(fit(cfg, hp, data, TV, {T: 1, V: 1}, x3['ft_epochs'], s, init=init, lr_scale=.3)[0], cfg, dt))
    print(f'neural done ({time.time() - t0:.0f}s)', flush=True)
    Xa = np.r_[Xt, Xv]; ya = np.r_[yt, yv]; Xte = tab_X(dt)
    # joint: same weighting rule as the neural X2 — target samples carry --ant-weight of the total weight
    wj = np.r_[np.full(len(yo), (1 - a.ant_weight) / len(yo)), np.full(len(ya), a.ant_weight / len(ya))]; wj *= len(wj) / wj.sum()
    for tag, (X, Y, W) in {'ant_only': (Xa, ya, None), 'others_only': (Xo, yo, None), 'joint': (np.r_[Xo, Xa], np.r_[yo, ya], wj)}.items():
        runs[f'LR_{tag}'] = [LogisticRegression(C=.03, class_weight='balanced', max_iter=5000).fit(X, Y, sample_weight=W).predict_proba(Xte)[:, 1]]
        runs[f'RF_{tag}'] = [RandomForestClassifier(500, min_samples_leaf=10, class_weight='balanced_subsample', random_state=s,
                                                    n_jobs=2).fit(X, Y, sample_weight=W).predict_proba(Xte)[:, 1] for s in seeds[:5]]

    rank_ens = {k: np.mean([rank01(r) for r in v], 0) for k, v in runs.items()}
    prob_ens = {k: np.mean(v, 0) for k, v in runs.items()}
    table = {}
    for k in runs:
        per = [{m: float(S.metric_fn(m)(r, y, loc)) for m in METRICS} for r in runs[k]]
        row = {m: [round(float(np.mean([p[m] for p in per])), 4), round(float(np.std([p[m] for p in per], ddof=1)) if len(per) > 1 else 0., 4)]
               for m in METRICS}
        for m in METRICS:
            e, lo, hi = S.bootstrap_ci(prob_ens[k] if 'density' in m else rank_ens[k], y, loc, m, a.boot)
            row[f'{m}_ens'] = [round(e, 4), round(lo, 4), round(hi, 4)]
        table[k] = row
    tests = []
    for k, ref in [('STAR_X1_others_only', 'STAR_X0_ant_only'), ('STAR_X2_joint', 'STAR_X0_ant_only'),
                   ('STAR_X3_pretrain_finetune', 'STAR_X0_ant_only'), ('LR_joint', 'LR_ant_only'), ('RF_joint', 'RF_ant_only'),
                   ('STAR_X2_joint', 'RF_joint'), ('STAR_X2_joint', 'LR_joint')]:
        for m in ['AP', 'Popt_density']:
            src = prob_ens if 'density' in m else rank_ens
            d_, lo, hi, p = S.paired_bootstrap(src[k], src[ref], y, loc, m, a.boot)
            tests.append({'model': k, 'vs': ref, 'metric': m, 'diff': round(d_, 4), 'CI95': [round(lo, 4), round(hi, 4)], 'p': round(p, 4)})
    rep = {'project': P, 'test': te, 'other_windows': O, 'n_other_classes': n_other, 'selection': sel, 'table': table, 'tests': tests,
           'seconds': round(time.time() - t0, 1)}
    (out / 'results.json').write_text(json.dumps(rep, indent=1))
    np.savez_compressed(out / 'test_scores.npz', ids=dt['ids'], y=y, loc=loc, **{f'rank_{k}': v for k, v in rank_ens.items()},
                        **{f'prob_{k}': v for k, v in prob_ens.items()})
    print(f'\n=== {P} {te} ===')
    for k, r in table.items():
        print(f"{k:28s} AP={r['AP'][0]:.3f}±{r['AP'][1]:.3f} ens {r['AP_ens'][0]:.3f} [{r['AP_ens'][1]:.3f},{r['AP_ens'][2]:.3f}]  "
              f"ROC={r['ROC_AUC'][0]:.3f}  Popt(dens)={r['Popt_density_ens'][0]:.3f}  R@20%LOC(dens)={r['R@20LOC_density_ens'][0]:.3f}")
    for t in tests:
        print(f"{t['model']:28s} vs {t['vs']:20s} {t['metric']:13s} Δ={t['diff']:+.4f} CI={t['CI95']} p={t['p']:.3f}")
    print(f'total {rep["seconds"]}s -> {out}')


if __name__ == '__main__':
    main()
