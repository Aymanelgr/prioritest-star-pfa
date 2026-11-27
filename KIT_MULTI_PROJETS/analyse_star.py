"""PRIORITEST-STAR — analyses du modèle final STAR-R sur le test (Ant 1.7), sans ré-entraînement.

Produit dans results/analysis/ :
  * budget_recall.pdf/png  : % de défauts trouvés en fonction du % de LOC inspecté (courbe effort/rappel) ;
  * fusion_weights.pdf/png : poids temporel alpha et accord cos(zT,zS) selon le label ;
  * ranked_classes.csv     : liste priorisée (rang, score, densité, label, LOC, alpha, accord, métriques clés) ;
  * analysis.json          : chiffres clés (Popt, rappels à budget fixé, erreurs typiques, stats alpha/accord).
Usage : python analyse_star.py --project ant
"""
import argparse, json, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd, torch
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from run_star import build_net, score_of, signed_log, rank01, releases_of  # noqa
from train import load  # noqa

BLUE, ORANGE, AQUA, GRAY, TXT, TXT2 = '#257de0', '#f47230', '#00b894', '#9a9a9a', '#191919', '#666666'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': '#bbbbbb', 'axes.labelcolor': TXT, 'xtick.color': TXT2,
                     'ytick.color': TXT2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': '#eeeeee', 'grid.linewidth': .8, 'legend.frameon': False})
KEY = ['loc', 'wmc', 'cbo', 'rfc', 'ca', 'ce', 'lcom', 'max_cc']


def effort_curve(order, y, loc):
    x = np.concatenate([[0], np.cumsum(loc[order])]) / loc.sum()
    r = np.concatenate([[0], np.cumsum(y[order])]) / max(1, y.sum())
    return x, r


def area(x, r): return float(np.trapezoid(r, x)) if hasattr(np, 'trapezoid') else float(np.trapz(r, x))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--project', default='ant')
    p.add_argument('--model', default=None); p.add_argument('--out', default=str(ROOT / 'results' / 'analysis'))
    a = p.parse_args(); torch.set_num_threads(2)
    rel = releases_of(a.project); te = rel[-1]
    mdir = ROOT / 'results' / 'star_r' / a.project
    ck = torch.load(a.model or mdir / 'star_r_ensemble.pt', weights_only=False)
    fm = json.loads((mdir / 'final_metrics.json').read_text())
    comp = {'STAR_R': ['STAR'], 'STAR_R+LR': ['STAR', 'LR'], 'STAR_R+ET': ['STAR', 'ET'],
            'STAR_R+LR+ET': ['STAR', 'LR', 'ET']}[fm['groups'][0]['chosen_on_validation']]
    d = load(ROOT / 'data' / a.project / f'{a.project}-{te}.npz', 'cpu'); d['x'] = signed_log(d['x'])
    y = d['y'].numpy().astype(int); loc = np.maximum(d['loc'].astype(float), 1); n = len(y)

    # STAR : moyenne des 10 réseaux (rangs), poids de fusion et accord moyens
    cfg = ck['config']; ranks, alpha, agree = [], [], []
    for sd in ck['state_dicts']:
        net = build_net(cfg, ck['n_features'], 'cpu'); net.load_state_dict(sd); net.eval()
        with torch.no_grad():
            o = net(d['x'], d['edges'])
        ranks.append(rank01(score_of(o, cfg).numpy())); alpha.append(o['gate'][:, 0].numpy()); agree.append(o['agreement'].numpy())
    parts = {'STAR': np.mean(ranks, 0)}
    tab = joblib.load(mdir / 'tabular_models.joblib')
    xt = d['x'].numpy(); X = np.c_[xt[:, -1], xt[:, -1] - xt[:, -2]]
    for k, m in tab.items(): parts[k] = rank01(m.predict_proba(X)[:, 1])
    score = np.mean([parts[k] for k in comp], 0); alpha = np.mean(alpha, 0); agree = np.mean(agree, 0)

    # Références : boosting A1 du kit (graine 42) si disponible
    curves = {}
    order_risk = np.argsort(-score, kind='stable'); order_dens = np.argsort(-score / loc, kind='stable')
    curves['STAR-R (tri par risque)'] = effort_curve(order_risk, y, loc)
    curves['STAR-R (tri par densité risque/LOC)'] = effort_curve(order_dens, y, loc)
    a1 = ROOT / 'results' / 'baseline_default' / a.project / 'seed42' / 'A1_boosting_scores.csv'
    if a1.exists():
        b = pd.read_csv(a1); b = b[b.split == 'test'].set_index('class_id').loc[d['ids']]
        curves['Boosting A1 (tri par risque)'] = effort_curve(np.argsort(-b['score'].values, kind='stable'), y, loc)
    rng = np.random.default_rng(0)
    rx = np.linspace(0, 1, 101); rr = np.mean([np.interp(rx, *effort_curve(rng.permutation(n), y, loc)) for _ in range(200)], 0)
    curves['Aléatoire'] = (rx, rr)
    opt = effort_curve(np.lexsort((loc, -y)), y, loc); worst = effort_curve(np.lexsort((-loc, y)), y, loc)
    popt = {k: round(1 - (area(*opt) - area(*c)) / (area(*opt) - area(*worst)), 4) for k, c in curves.items()}
    at = lambda c, b: float(np.interp(b, *c))
    budgets = {k: {f'{int(b*100)}%LOC': round(at(c, b), 4) for b in (.1, .2, .3, .5)} for k, c in curves.items()}

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    # Figure 1 : effort / rappel
    style = {'STAR-R (tri par risque)': (BLUE, '-'), 'STAR-R (tri par densité risque/LOC)': (ORANGE, '-'),
             'Boosting A1 (tri par risque)': (AQUA, '-'), 'Aléatoire': (GRAY, '--')}
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(opt[0] * 100, opt[1] * 100, color='#bbbbbb', lw=1.4, ls=':', label='Optimal (oracle)')
    for k, (x, r) in curves.items():
        c, ls = style[k]; ax.plot(x * 100, r * 100, color=c, lw=2, ls=ls, label=f'{k}  (Popt {popt[k]:.2f})')
    ax.axvline(20, color='#bbbbbb', lw=1, ls='--'); ax.text(21, 3, 'budget 20 % LOC', color=TXT2, fontsize=8.5)
    ax.set_xlabel('Lignes de code inspectées (%)'); ax.set_ylabel('Classes défectueuses trouvées (%)')
    ax.set_xlim(0, 100); ax.set_ylim(0, 102); ax.legend(fontsize=8, loc='lower right')
    ax.set_title(f'Courbe effort–rappel, {a.project.capitalize()} {te} (test)', fontsize=11, color=TXT, loc='left')
    fig.tight_layout(); [fig.savefig(out / f'budget_recall.{e}', dpi=200) for e in ('pdf', 'png')]; plt.close(fig)

    # Figure 2 : poids de fusion et accord
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.2))
    for ax, v, lab in ((axs[0], alpha, r'Poids temporel $\alpha$ ($\beta = 1-\alpha$)'), (axs[1], agree, r'Accord $\cos(z_T, z_S)$')):
        bins = np.linspace(np.quantile(v, .005), v.max(), 16)
        ax.hist([np.clip(v[y == 0], bins[0], None), np.clip(v[y == 1], bins[0], None)], bins=bins, color=[BLUE, ORANGE],
                label=[f'Saines (n={int((y==0).sum())})', f'Défectueuses (n={int(y.sum())})'], rwidth=.9)
        ax.set_xlabel(lab); ax.set_ylabel('Nombre de classes'); ax.grid(axis='x', visible=False)
    axs[0].legend(fontsize=8)
    fig.tight_layout(); [fig.savefig(out / f'fusion_weights.{e}', dpi=200) for e in ('pdf', 'png')]; plt.close(fig)

    # Tableau priorisé avec métriques brutes
    g = np.load(ROOT / 'data' / a.project / f'{a.project}-{te}_graph_temporal.npz')
    feats = json.loads((ROOT / 'data' / 'manifest_multi.json').read_text())['features']
    raw = g['features_raw_v2'][g['target_indices_last']]
    rank = np.empty(n, int); rank[order_risk] = np.arange(1, n + 1)
    rankd = np.empty(n, int); rankd[order_dens] = np.arange(1, n + 1)
    df = pd.DataFrame({'rang_risque': rank, 'rang_densite': rankd, 'classe': d['ids'], 'defectueuse': y,
                       'score': score.round(4), 'risque_par_kLOC': (1000 * score / loc).round(4),
                       'alpha_temporel': alpha.round(4), 'accord_cos': agree.round(4),
                       **{m: raw[:, feats.index(m)] for m in KEY}}).sort_values('rang_risque')
    df.to_csv(out / 'ranked_classes.csv', index=False)

    k = int(round(fm['groups'][0]['decision_flag_rate'] * n)); flagged = set(order_risk[:k])
    fp = df[(df.rang_risque <= k) & (df.defectueuse == 0)].head(5)
    fn = df[(df.rang_risque > k) & (df.defectueuse == 1)].tail(5)
    s = lambda v: {'moyenne': round(float(v.mean()), 4), 'ecart_type': round(float(v.std()), 4)}
    res = {'project': a.project, 'test': te, 'blend': comp, 'n': n, 'positives': int(y.sum()),
           'Popt': popt, 'recall_at_LOC_budget': budgets, 'flagged_k': k,
           'alpha_temporal': {'sains': s(alpha[y == 0]), 'defectueux': s(alpha[y == 1])},
           'agreement_cos': {'sains': s(agree[y == 0]), 'defectueux': s(agree[y == 1])},
           'median_LOC': {'sains': float(np.median(loc[y == 0])), 'defectueux': float(np.median(loc[y == 1])),
                          'faux_positifs': float(np.median(loc[[i for i in flagged if y[i] == 0]])),
                          'defauts_manques': float(np.median(loc[[i for i in range(n) if y[i] == 1 and i not in flagged]]))},
           'top_false_positives': fp[['rang_risque', 'classe', 'loc', 'wmc', 'cbo', 'rfc']].to_dict('records'),
           'worst_missed_defects': fn[['rang_risque', 'classe', 'loc', 'wmc', 'cbo', 'rfc']].to_dict('records')}
    (out / 'analysis.json').write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float))
    print(json.dumps({k: res[k] for k in ['Popt', 'recall_at_LOC_budget', 'alpha_temporal', 'agreement_cos', 'median_LOC']}, indent=1, ensure_ascii=False))
    print(fp.to_string(index=False)); print(fn.to_string(index=False)); print('->', out)


if __name__ == '__main__':
    main()
