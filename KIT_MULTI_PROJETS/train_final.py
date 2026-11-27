"""PRIORITEST-STAR — entraînement final « STAR-R » (robuste) et évaluation unique sur le test.

Toutes les décisions sont prises sur la VALIDATION (fenêtre 2) avec des modèles entraînés sur la fenêtre 1 :
  * hyperparamètres + nombre d'époques : lus dans config_star_r.json (produit par explore_val.py) ;
  * composition de l'ensemble final (STAR seul / STAR + tabulaire) : meilleure AP de validation ;
  * règle de décision (fraction de classes signalées) : meilleure accuracy de validation.
Ensuite les modèles sont ré-entraînés sur fenêtres 1+2 avec les mêmes réglages, et le test (fenêtre 3)
est évalué UNE fois. Répété sur plusieurs « groupes » de graines pour mesurer la dispersion.

Usage : python train_final.py --project ant --groups 5 --ens 10
"""
import argparse, json, sys, time
from pathlib import Path
import joblib, numpy as np, torch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from run_star import (VARIANTS, build_net, cohort_loss, score_of, set_seed, signed_log, releases_of,  # noqa
                      rank01, evaluate, export_scores)
from train import load, choose_flag_rate  # noqa


def train_net(cfg, hp, data, keys, epochs, seed):
    set_seed(seed)
    net = build_net(cfg, data[keys[0]]['x'].shape[-1], 'cpu')
    opt = torch.optim.Adam(net.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay'])
    for _ in range(epochs):
        net.train(); opt.zero_grad()
        loss = sum(cohort_loss(net, cfg, hp, data[k], True) for k in keys) / len(keys)
        loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
    net.eval(); return net


def star_scores(nets, cfg, d):
    with torch.no_grad():
        return np.mean([rank01(score_of(n(d['x'], d['edges']), cfg).numpy()) for n in nets], axis=0)


def tab_X(d):
    x = d['x'].numpy(); return np.c_[x[:, -1], x[:, -1] - x[:, -2]]   # dernière version + Δ (déjà signed_log)


def fit_tab(data, keys, seed, C):
    X = np.concatenate([tab_X(data[k]) for k in keys]); y = np.concatenate([data[k]['y'].numpy() for k in keys])
    lr = LogisticRegression(C=C, class_weight='balanced', max_iter=5000).fit(X, y)
    et = ExtraTreesClassifier(500, min_samples_leaf=3, class_weight='balanced_subsample', random_state=seed, n_jobs=2).fit(X, y)
    return {'LR': lr, 'ET': et}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--project', default='ant'); p.add_argument('--config', default=str(ROOT / 'config_star_r.json'))
    p.add_argument('--groups', type=int, default=5, help='répétitions indépendantes de toute la procédure')
    p.add_argument('--ens', type=int, default=10, help='graines par ensemble STAR')
    p.add_argument('--out', default=str(ROOT / 'results' / 'star_r'))
    a = p.parse_args(); torch.set_num_threads(2)
    conf = json.loads(Path(a.config).read_text())
    cfg = {**VARIANTS['A6_STAR_full'], **conf['model']}; hp = conf['hp']; E = conf['epochs']; C = conf['lr_C']
    rel = releases_of(a.project); assert len(rel) >= 3
    tr, val, te = rel
    data = {v: load(ROOT / 'data' / a.project / f'{a.project}-{v}.npz', 'cpu') for v in rel}
    for d in data.values(): d['x'] = signed_log(d['x'])
    out = Path(a.out) / a.project; out.mkdir(parents=True, exist_ok=True)
    yv = data[val]['y'].numpy()
    candidates = {'STAR_R': ['STAR'], 'STAR_R+LR': ['STAR', 'LR'], 'STAR_R+ET': ['STAR', 'ET'],
                  'STAR_R+LR+ET': ['STAR', 'LR', 'ET'], 'LR': ['LR'], 'ET': ['ET']}
    groups = []; t_all = time.time()
    for g in range(a.groups):
        t0 = time.time(); seeds = [1000 * (g + 1) + i for i in range(a.ens)]
        # 1) modèles de SÉLECTION (train seulement) -> scores de validation
        sel = {'STAR': star_scores([train_net(cfg, hp, data, [tr], E, s) for s in seeds], cfg, data[val])}
        tab = fit_tab(data, [tr], seeds[0], C)
        for k, m in tab.items(): sel[k] = rank01(m.predict_proba(tab_X(data[val]))[:, 1])
        val_ap = {c: average_precision_score(yv, np.mean([sel[k] for k in comp], axis=0)) for c, comp in candidates.items()}
        # le choix n'est fait que parmi les variantes contenant STAR (exigence de la fiche) ; LR/ET = références
        choice = max([c for c in candidates if 'STAR' in candidates[c]], key=val_ap.get)
        sv = np.mean([sel[k] for k in candidates[choice]], axis=0)
        rate = choose_flag_rate(sv, yv)
        # 2) ré-entraînement train+validation, mêmes réglages ; 3) test évalué une fois
        nets = [train_net(cfg, hp, data, [tr, val], E, s) for s in seeds]
        fin = {'STAR': star_scores(nets, cfg, data[te])}
        tabf = fit_tab(data, [tr, val], seeds[0], C)
        for k, m in tabf.items(): fin[k] = rank01(m.predict_proba(tab_X(data[te]))[:, 1])
        res = {'group': g, 'seeds': seeds, 'chosen_on_validation': choice, 'decision_flag_rate': round(rate, 4),
               'validation_AP_by_candidate': {k: round(v, 4) for k, v in val_ap.items()}, 'test': {}}
        for c, comp in candidates.items():
            e = {}; evaluate(e, 'test', np.mean([fin[k] for k in comp], axis=0), data[te],
                             choose_flag_rate(np.mean([sel[k] for k in comp], axis=0), yv))
            res['test'][c] = e['test']
        res['test']['FINAL'] = res['test'][choice]
        res['seconds'] = round(time.time() - t0, 1)
        if g == 0:   # sauvegarde des modèles finaux du premier groupe + scores
            torch.save({'config': cfg, 'hp': hp, 'epochs': E, 'seeds': seeds, 'n_features': data[tr]['x'].shape[-1],
                        'preprocessing': 'signed_log(z-score fit on first window)', 'trained_on': [tr, val],
                        'state_dicts': [n.state_dict() for n in nets]}, out / 'star_r_ensemble.pt')
            joblib.dump(tabf, out / 'tabular_models.joblib')
            sc = out / 'final_test_scores.csv'; sc.unlink(missing_ok=True)
            export_scores(sc, 'test', data[te], np.mean([fin[k] for k in candidates[choice]], axis=0))
        groups.append(res); t = res['test']['FINAL']
        print(f'[group {g}] choix(val)={choice} valAP={val_ap[choice]:.4f} | TEST AP={t["AP"]:.4f} '
              f'Acc={t["Accuracy"]:.4f} F1={t["F1"]:.4f} MCC={t["MCC"]:.4f} R@20%LOC={t["Recall@20pct_LOC"]:.4f} '
              f'EA_R@20%LOC={t["EA_Recall@20pct_LOC"]:.4f} ({res["seconds"]}s)', flush=True)
    summ = {}
    for c in list(candidates) + ['FINAL']:
        summ[c] = {m: [round(float(np.mean([gr['test'][c][m] for gr in groups])), 4),
                       round(float(np.std([gr['test'][c][m] for gr in groups], ddof=1)), 4) if len(groups) > 1 else 0.]
                   for m in groups[0]['test'][c] if isinstance(groups[0]['test'][c][m], (int, float))}
    report = {'project': a.project, 'train': tr, 'validation': val, 'test': te, 'config': conf,
              'protocol': 'decisions on validation (train-only models); refit train+val; single test evaluation',
              'total_seconds': round(time.time() - t_all, 1), 'groups': groups, 'summary_mean_std': summ}
    (out / 'final_metrics.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print('\n=== TEST (moyenne ± écart-type sur', a.groups, 'groupes) ===')
    for c, s in summ.items():
        print(f'{c:14s} ' + '  '.join(f'{m}={s[m][0]:.4f}±{s[m][1]:.4f}' for m in
                                     ['AP', 'ROC_AUC', 'Accuracy', 'Balanced_accuracy', 'Precision', 'Recall', 'F1', 'MCC',
                                      'Recall@20pct_LOC', 'EA_Recall@20pct_LOC']))
    print('total', report['total_seconds'], 's ->', out)


if __name__ == '__main__':
    main()
