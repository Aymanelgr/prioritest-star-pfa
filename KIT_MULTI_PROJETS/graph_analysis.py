"""PRIORITEST-STAR — step 7.3 of ARCHITECTURE.md: why does the import graph help on Camel and not on Ant / POI?

For every project's TEST window it measures:
  * graph shape: classes, import edges, mean degree, share of isolated classes (no import in or out);
  * defect homophily: among connected classes, the share of defective neighbours of defective classes vs of clean
    classes (a graph can only help a GNN if defects cluster along its edges), plus the label assortativity coefficient;
  * where the graph helps: A2 (TCN, no graph) vs A3 (+ graph, classic fusion), both with the configs FROZEN from Ant
    (results/phase1), 5 seeds, AP computed separately on isolated and on connected classes.
Labels are only used for evaluation statistics on the test window, never to train anything here.
Usage: python graph_analysis.py
"""
import json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from run_star import signed_log, rank01, score_of  # noqa: E402
from train_final import train_net  # noqa: E402
from train import load  # noqa: E402
import stats as S  # noqa: E402

SPLITS = {'ant': (['1.5', '1.6'], '1.7'), 'camel': (['1.4'], '1.6'), 'poi': (['2.5'], '3.0'), 'jedit': (['4.1', '4.2'], '4.3')}


def graph_stats(d):
    y = d['y'].numpy().astype(int); e = d['edges'].numpy(); n = len(y)
    und = np.unique(np.sort(e, 1), axis=0) if len(e) else e
    deg = np.bincount(und.ravel(), minlength=n) if len(und) else np.zeros(n, int)
    nb = [[] for _ in range(n)]
    for a, b in und:
        nb[a].append(b); nb[b].append(a)
    conn = deg > 0
    nbr_rate = np.array([y[v].mean() if v else np.nan for v in nb])
    hom_def = float(np.nanmean(nbr_rate[(y == 1) & conn])) if ((y == 1) & conn).any() else float('nan')
    hom_clean = float(np.nanmean(nbr_rate[(y == 0) & conn])) if ((y == 0) & conn).any() else float('nan')
    # label assortativity (Newman) on the undirected edge list
    if len(und):
        a = np.r_[y[und[:, 0]], y[und[:, 1]]]; b = np.r_[y[und[:, 1]], y[und[:, 0]]]
        assort = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float('nan')
    else:
        assort = float('nan')
    return {'classes': n, 'defect_rate': round(float(y.mean()), 3), 'import_edges': int(len(und)),
            'mean_degree': round(float(deg.mean()), 2), 'isolated_share': round(float((~conn).mean()), 3),
            'defect_rate_isolated': round(float(y[~conn].mean()), 3) if (~conn).any() else None,
            'defect_rate_connected': round(float(y[conn].mean()), 3) if conn.any() else None,
            'neighbour_defect_rate_of_defective': round(hom_def, 3),
            'neighbour_defect_rate_of_clean': round(hom_clean, 3),
            'homophily_gap': round(hom_def - hom_clean, 3), 'label_assortativity': round(assort, 3)}, conn


def main():
    torch.set_num_threads(2)
    t1 = json.loads((ROOT / 'results' / 'phase1' / 'ant' / 'tuning.json').read_text())
    out = ROOT / 'results' / 'graph_analysis'; out.mkdir(parents=True, exist_ok=True); res = {}
    for P, (trs, te) in SPLITS.items():
        data = {v: load(ROOT / 'data' / P / f'{P}-{v}.npz', 'cpu') for v in trs + [te]}
        for d in data.values():
            d['x'] = signed_log(d['x'])
        dt = data[te]; st, conn = graph_stats(dt); y = dt['y'].numpy().astype(int)
        sc = {}
        for n in ['A2_TCN_BCE', 'A3_graph_classic_fusion_BCE']:
            t = t1[n]; runs = []
            for s in range(100, 105):
                net = train_net(t['cfg'], t['hp'], data, trs, t['epochs'], s)
                with torch.no_grad():
                    runs.append(score_of(net(dt['x'], dt['edges']), t['cfg']).numpy())
            sc[n] = np.mean([rank01(r) for r in runs], 0)
        by = {}
        for name, m in [('all', np.ones_like(conn)), ('isolated', ~conn), ('connected', conn)]:
            if m.sum() and 0 < y[m].sum() < m.sum():
                a2 = S.metric_fn('AP')(sc['A2_TCN_BCE'][m], y[m], None); a3 = S.metric_fn('AP')(sc['A3_graph_classic_fusion_BCE'][m], y[m], None)
                by[name] = {'n': int(m.sum()), 'defective': int(y[m].sum()), 'AP_A2_no_graph': round(float(a2), 4),
                            'AP_A3_graph': round(float(a3), 4), 'gain_from_graph': round(float(a3 - a2), 4)}
        res[P] = {'test': te, 'graph': st, 'AP_by_degree': by}
        print(f"\n== {P} {te}: {st}")
        for k, v in by.items():
            print(f"   {k:10s} n={v['n']:4d} def={v['defective']:3d}  AP A2={v['AP_A2_no_graph']:.3f}  A3={v['AP_A3_graph']:.3f}  Δ={v['gain_from_graph']:+.3f}")
    (out / 'graph_analysis.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
