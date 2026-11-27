"""PRIORITEST-STAR — step 6.3 of ARCHITECTURE.md: do the Ant findings hold on other projects?

Every setting is FROZEN from the Ant studies (results/phase1, phase3, phase4): architecture, hyper-parameters,
number of epochs, tabular parameters, blend recipe. Nothing is tuned on the projects below, so their test releases
are genuinely held out (R2).
  camel 1.4 -> 1.6, poi 2.5 -> 3.0 : train on the first window, test on the second (no validation window exists)
  jedit 4.1, 4.2 -> 4.3            : train on 4.1 + 4.2, test on 4.3 (only 5 defective classes: CI reported, caveat)
Claims checked, per project:
  C1  STAR variants ≈ LR on AP                     (Phase 1)
  C2  tree ensembles beat LR on Popt (score / LOC)  (Phase 1)
  C3  the STAR + LR + ET blend beats LR on Popt     (Phase 1)
  C4  joint cross-project training hurts LR more than STAR (Phase 3); 'others' = every other project, never the target
Usage: python transfer_study.py            (all three projects, ~6 min)
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from run_star import signed_log, rank01, score_of  # noqa: E402
from compare_fair import TABULAR, evaluate, METRICS  # noqa: E402
from train_final import train_net, tab_X  # noqa: E402
from more_data import windows, fit, scores  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from train import load  # noqa: E402
import stats as S  # noqa: E402

SPLITS = {'camel': (['1.4'], '1.6'), 'poi': (['2.5'], '3.0'), 'jedit': (['4.1', '4.2'], '4.3')}
NEURAL = ['A2_TCN_BCE', 'A3_graph_classic_fusion_BCE', 'A4_similarity_fusion_BCE', 'A5_plus_InfoNCE', 'A6_STAR_full']


def run_project(P, seeds, boot, frozen, joint_epochs, ant_weight):
    trs, te = SPLITS[P]
    data = {v: load(ROOT / 'data' / P / f'{P}-{v}.npz', 'cpu') for v in trs + [te]}
    for d in data.values():
        d['x'] = signed_log(d['x'])
    dt = data[te]; y = dt['y'].numpy().astype(int); loc = dt['loc']
    Xtr = np.concatenate([tab_X(data[v]) for v in trs]); ytr = np.concatenate([data[v]['y'].numpy() for v in trs]); Xte = tab_X(dt)
    runs = {}
    for n, t in frozen.items():
        if 'cfg' in t:
            runs[n] = []
            for s in seeds:
                net = train_net(t['cfg'], t['hp'], data, trs, t['epochs'], s)
                with torch.no_grad():
                    runs[n].append(score_of(net(dt['x'], dt['edges']), t['cfg']).numpy())
    for n in ['A1_HGB', 'B_LR', 'B_RF', 'B_ET']:
        runs[n] = [TABULAR[n](frozen[n]['params'], s).fit(Xtr, ytr).predict_proba(Xte)[:, 1] for s in (seeds if n != 'B_LR' else seeds[:1])]
    # joint cross-project (C4): STAR A5 and LR, other projects = every project except P (Ant included as a source)
    others = windows(P); allw = {**{f'{P}-{v}': data[v] for v in trs}, **others}
    tk = [f'{P}-{v}' for v in trs]; ok = list(others); tot = sum(len(others[k]['y']) for k in ok)
    w = {**{k: ant_weight / len(tk) for k in tk}, **{k: (1 - ant_weight) * len(others[k]['y']) / tot for k in ok}}
    a5 = frozen['A5_plus_InfoNCE']
    runs['STAR_A5_joint'] = [scores(fit(a5['cfg'], a5['hp'], allw, tk + ok, w, joint_epochs, s)[0], a5['cfg'], dt) for s in seeds]
    Xo = np.concatenate([tab_X(others[k]) for k in ok]); yo = np.concatenate([others[k]['y'].numpy() for k in ok])
    sw = np.r_[np.full(len(yo), (1 - ant_weight) / len(yo)), np.full(len(ytr), ant_weight / len(ytr))]; sw *= len(sw) / sw.sum()
    runs['B_LR_joint'] = [LogisticRegression(C=frozen['B_LR']['params']['C'], class_weight='balanced', max_iter=5000)
                          .fit(np.r_[Xo, Xtr], np.r_[yo, ytr], sample_weight=sw).predict_proba(Xte)[:, 1]]

    ens = {k: np.mean([rank01(r) for r in v], 0) for k, v in runs.items()}
    prob = {k: np.mean(v, 0) for k, v in runs.items()}
    ens['STAR_R_blend'] = np.mean([ens['A6_STAR_full'], rank01(ens['B_LR']), rank01(ens['B_ET'])], 0)
    prob['STAR_R_blend'] = np.mean([prob['A6_STAR_full'], prob['B_LR'], prob['B_ET']], 0)
    table = {}
    for k in ens:
        per = [evaluate(r, dt) for r in runs.get(k, [prob[k]])]
        row = {m: round(float(np.mean([p[m] for p in per])), 4) for m in METRICS}
        row['AP_std'] = round(float(np.std([p['AP'] for p in per], ddof=1)), 4) if len(per) > 1 else 0.
        for m in ['AP', 'Popt_density']:
            e, lo, hi = S.bootstrap_ci(prob[k] if 'density' in m else ens[k], y, loc, m, boot)
            row[f'{m}_ens'] = [round(e, 4), round(lo, 4), round(hi, 4)]
        table[k] = row
    tests = []
    for k, ref, ms in ([(n, 'B_LR', ['AP', 'Popt_density']) for n in NEURAL + ['G2_gate_entropy']] +
                       [('B_RF', 'B_LR', ['AP', 'Popt_density']), ('B_ET', 'B_LR', ['AP', 'Popt_density']),
                        ('STAR_R_blend', 'B_LR', ['AP', 'Popt_density']),
                        ('STAR_A5_joint', 'A5_plus_InfoNCE', ['AP']), ('B_LR_joint', 'B_LR', ['AP']),
                        ('STAR_A5_joint', 'B_LR_joint', ['AP'])]):
        for m in ms:
            src = prob if 'density' in m else ens
            d_, lo, hi, p = S.paired_bootstrap(src[k], src[ref], y, loc, m, boot)
            tests.append({'model': k, 'vs': ref, 'metric': m, 'diff': round(d_, 4), 'CI95': [round(lo, 4), round(hi, 4)], 'p': round(p, 4)})
    return {'project': P, 'train': trs, 'test': te, 'n_test': int(len(y)), 'defective_test': int(y.sum()),
            'table': table, 'tests': tests}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--projects', nargs='+', default=['camel', 'poi', 'jedit']); ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--boot', type=int, default=2000); ap.add_argument('--out', default=str(ROOT / 'results' / 'transfer'))
    a = ap.parse_args(); torch.set_num_threads(2); t0 = time.time()
    p1 = json.loads((ROOT / 'results' / 'phase1' / 'ant' / 'tuning.json').read_text())
    p4 = json.loads((ROOT / 'results' / 'phase4' / 'ant' / 'tuning.json').read_text())
    p3 = json.loads((ROOT / 'results' / 'phase3' / 'ant' / 'selection.json').read_text())
    frozen = {**{n: p1[n] for n in NEURAL}, 'G2_gate_entropy': p4['G2_gate_entropy'],
              **{n: p1[n] for n in ['A1_HGB', 'B_LR', 'B_RF', 'B_ET']}}
    seeds = [100 + i for i in range(a.seeds)]; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    allres = {}
    for P in a.projects:
        t1 = time.time(); r = run_project(P, seeds, a.boot, frozen, p3['X2_joint']['epochs'], .5); r['seconds'] = round(time.time() - t1, 1)
        allres[P] = r; (out / f'{P}.json').write_text(json.dumps(r, indent=1))
        print(f"\n=== {P}: train {r['train']} -> test {r['test']} ({r['n_test']} classes, {r['defective_test']} defective), {r['seconds']}s ===")
        for k, v in r['table'].items():
            print(f"{k:28s} AP={v['AP']:.3f}±{v['AP_std']:.3f} ens {v['AP_ens'][0]:.3f} [{v['AP_ens'][1]:.3f},{v['AP_ens'][2]:.3f}] "
                  f"ROC={v['ROC_AUC']:.3f} Popt(dens)={v['Popt_density_ens'][0]:.3f} R@20%LOC(dens)={v['R@20LOC_density']:.3f}")
        for t in r['tests']:
            print(f"  {t['model']:26s} vs {t['vs']:16s} {t['metric']:13s} Δ={t['diff']:+.4f} CI={t['CI95']} p={t['p']:.3f}")
    print(f'total {time.time() - t0:.0f}s -> {out}')


if __name__ == '__main__':
    main()
