"""PRIORITEST-STAR — lanceur complet des ablations (fiche PFA §6) + variante améliorée STAR+ + ensemble.

Exemples :
  python run_star.py --project ant --seeds 42 43 44 45 46 --tune
  python run_star.py --project ant --seeds 42 43 44 45 46 --tune --improved
  python run_star.py --project ant --variants A2_TCN_BCE A6_STAR_full --epochs 20

Protocole :
  3 fenêtres cibles : 1re = entraînement, 2e = validation (early stopping + réglages), 3e = test final.
  2 fenêtres       : 1re = entraînement, 2e = test exploratoire ; nombre d'époques fixe, aucun réglage.
  Détection des défauts de la version observée — PAS une prédiction de la version suivante.
  Le test n'est JAMAIS utilisé pour régler quoi que ce soit (époque, hyperparamètres, seuil).

--improved active, pour TOUTES les variantes (comparaison équitable) :
  1. réentraînement sur entraînement + validation après sélection (époque, hyperparamètres) — test intact ;
  2. pos_weight dans la BCE (rééquilibrage des classes défectueuses) ;
  3. compression des queues lourdes : sign(x)·log(1+|x|) sur les métriques standardisées.
Toujours actifs : règle de décision (fraction de classes signalées) réglée sur la validation → Accuracy,
Balanced accuracy, MCC, F1 ; ensemble A8 = moyenne des rangs de A1 (boosting), A5 et A6.

Sorties (results/<projet>[_improved]/) :
  seed<k>/metrics.json, seed<k>/<variante>_scores.csv, summary.csv, tuning.json (si --tune)
"""
import argparse, copy, csv, itertools, json, random, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from model import (CrossViewSTAR, cross_view_infonce, label_aware_infonce, supcon_cross_view, gate_entropy,  # noqa: E402
                   pairwise_rank_loss, effort_aware_rank_loss)
from train import load, stats, effort_aware_stats, choose_flag_rate, classification_stats  # noqa: E402

# Chaque ligne ajoute UN mécanisme à la précédente (A1..A6), A7 = améliorations proposées, A8 = ensemble.
VARIANTS = {
    'A1_boosting': None,
    'A2_TCN_BCE': dict(use_graph=False, similarity_gate=False, lambda_c=0., lambda_r=0.),
    'A3_graph_classic_fusion_BCE': dict(use_graph=True, similarity_gate=False, lambda_c=0., lambda_r=0.),
    'A4_similarity_fusion_BCE': dict(use_graph=True, similarity_gate=True, lambda_c=0., lambda_r=0.),
    'A5_plus_InfoNCE': dict(use_graph=True, similarity_gate=True, lambda_c=.25, lambda_r=0.),
    'A6_STAR_full': dict(use_graph=True, similarity_gate=True, lambda_c=.25, lambda_r=.25),
    'A7_STAR_plus': dict(use_graph=True, similarity_gate=True, lambda_c=.25, lambda_r=.25,
                         use_delta=True, bidirectional=True, dropout=.1,
                         label_aware=True, effort_aware=True),
    'A8_ensemble': 'ensemble',
}
ENSEMBLE_OF = ['A1_boosting', 'A5_plus_InfoNCE', 'A6_STAR_full']   # fixé a priori, pas choisi sur le test
DEFAULT_HP = dict(temperature=.2, lr=2e-3, weight_decay=1e-4)
GRID = {'lambda_c': [.1, .25, .5], 'temperature': [.1, .2, .5], 'lambda_r': [.25, .5]}


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def releases_of(project):
    m = json.loads((ROOT / 'data' / 'manifest_multi.json').read_text(encoding='utf-8'))
    info = m['projects'].get(project)
    if not info:
        raise SystemExit(f"Projet inconnu : {project}. Disponibles : {', '.join(m['projects'])}")
    rel = [t['version'] for t in info['targets'] if t['status'] == 'prepared']
    if len(rel) < 2:
        raise SystemExit(f"{project} : {len(rel)} fenêtre(s) seulement — pas de découpage chronologique train/test possible.")
    return rel


def signed_log(x):
    """Compression des queues lourdes (LOC, RFC, WMC… très asymétriques) ; monotone, appliquée à toutes les versions."""
    return torch.sign(x) * torch.log1p(x.abs())


def check_finite(out, where):
    for k, v in out.items():
        if torch.is_tensor(v) and not torch.isfinite(v).all():
            raise FloatingPointError(f"NaN/Inf dans '{k}' ({where})")


def score_of(out, cfg):
    return torch.sigmoid(out['rank'] if cfg['lambda_r'] else out['logits'])


def build_net(cfg, n_features, device):
    return CrossViewSTAR(n_features, use_graph=cfg['use_graph'], similarity_gate=cfg['similarity_gate'],
                         use_delta=cfg.get('use_delta', False), bidirectional=cfg.get('bidirectional', False),
                         dropout=cfg.get('dropout', 0.), gate_absdiff=cfg.get('gate_absdiff', False)).to(device)


def cohort_loss(net, cfg, hp, a, pos_weight):
    """Perte sur UNE cohorte (une version, son graphe). RankNet compare des classes d'une même version."""
    out = net(a['x'], a['edges'])
    check_finite(out, 'forward')
    pw = None
    if pos_weight:
        pos = a['y'].sum().clamp_min(1)
        pw = ((len(a['y']) - pos) / pos).reshape(1)
    loss = F.binary_cross_entropy_with_logits(out['logits'], a['y'], pos_weight=pw)
    if cfg['lambda_r']:
        loc_t = torch.as_tensor(a['loc'], device=a['y'].device, dtype=torch.float32)
        rl = effort_aware_rank_loss(out['rank'], a['y'], loc_t) if cfg.get('effort_aware') \
            else pairwise_rank_loss(out['rank'], a['y'])
        loss = loss + cfg['lambda_r'] * rl
    if cfg['lambda_c']:
        if cfg.get('contrast') == 'supcon':
            cl = supcon_cross_view(out['temporal'], out['structural'], a['y'], hp['temperature'], cfg.get('sup_weight', .5))
        else:
            cl = label_aware_infonce(out['temporal'], out['structural'], a['y'], hp['temperature']) if cfg.get('label_aware') \
                else cross_view_infonce(out['temporal'], out['structural'], hp['temperature'])
        loss = loss + cfg['lambda_c'] * cl
    if cfg.get('gate_entropy') and cfg['use_graph']:
        loss = loss - cfg['gate_entropy'] * gate_entropy(out['gate'])
    return loss


def train_variant(cfg, hp, data, train_keys, val, epochs, patience, device, min_epochs=10, pos_weight=False):
    """Entraîne une variante neuronale (batch complet par version). Avec val=None : exactement `epochs` époques.

    min_epochs : aucune sélection de checkpoint ni arrêt anticipé avant cette époque. Sans ce plancher, une AP de
    validation bruitée peut retenir un modèle quasi non entraîné (époque 1), ce qui rend les ablations instables.
    """
    min_epochs = max(1, min(min_epochs, epochs))
    net = build_net(cfg, data[train_keys[0]]['x'].shape[-1], device)
    opt = torch.optim.Adam(net.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay'])
    best, best_state, best_ep, wait = -1., None, epochs, 0
    for ep in range(1, epochs + 1):
        net.train(); opt.zero_grad()
        loss = sum(cohort_loss(net, cfg, hp, data[k], pos_weight) for k in train_keys) / len(train_keys)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'perte non finie à l\'époque {ep}')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        if val is None or ep < min_epochs:
            continue
        net.eval()
        with torch.no_grad():
            s = score_of(net(data[val]['x'], data[val]['edges']), cfg).cpu().numpy()
        ap = average_precision_score(data[val]['y'].cpu().numpy(), s)
        if ap > best:
            best, best_ep, wait = ap, ep, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in net.state_dict().items()})
        else:
            wait += 1
            if patience and wait >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net, best_ep, best


def export_scores(path, split, d, scores, out=None):
    new = not path.exists()
    with path.open('a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['split', 'class_id', 'label', 'loc', 'score', 'risk_per_kloc',
                        'alpha_temporal', 'beta_structural', 'agreement_cos'])
        y = d['y'].cpu().numpy()
        for i in range(len(y)):
            alpha = beta = agr = ''
            if out is not None:
                alpha = round(float(out['gate'][i, 0]), 5); beta = round(float(out['gate'][i, 1]), 5)
                agr = round(float(out['agreement'][i]), 5)
            density = 1000 * float(scores[i]) / max(float(d['loc'][i]), 1.)
            w.writerow([split, d['ids'][i], int(y[i]), float(d['loc'][i]), round(float(scores[i]), 6),
                        round(density, 6), alpha, beta, agr])


def tune(data, tr, val, epochs, patience, device, seed, names, min_epochs=10, pos_weight=False):
    """Recherche en grille sur la VALIDATION uniquement (jamais sur le test)."""
    chosen = {}
    for name in names:
        cfg = VARIANTS[name]
        if not isinstance(cfg, dict) or (not cfg['lambda_c'] and not cfg['lambda_r']):
            continue
        keys = [k for k in GRID if (k != 'lambda_r' or cfg['lambda_r']) and (k == 'lambda_r' or cfg['lambda_c'])]
        best = (-1, None)
        for combo in itertools.product(*[GRID[k] for k in keys]):
            p = dict(zip(keys, combo))
            c = {**cfg, **{k: v for k, v in p.items() if k.startswith('lambda')}}
            hp = {**DEFAULT_HP, **{k: v for k, v in p.items() if k == 'temperature'}}
            set_seed(seed)
            _, _, ap = train_variant(c, hp, data, [tr], val, epochs, patience, device, min_epochs, pos_weight)
            print(f'  [tune] {name} {p} -> AP_val={ap:.4f}', flush=True)
            if ap > best[0]:
                best = (ap, p)
        chosen[name] = {'params': best[1], 'AP_validation': round(best[0], 4)}
        print(f'[tune] {name} retenu : {best[1]} (AP_val={best[0]:.4f})', flush=True)
    return chosen


def rank01(s):
    s = np.asarray(s, dtype=float)
    return s.argsort(kind='stable').argsort() / max(1, len(s) - 1)


def evaluate(entry, split, s, d, rate):
    y = d['y'].cpu().numpy()
    entry[split] = stats(s, y, d['loc'])
    entry[split].update(effort_aware_stats(s, y, d['loc']))
    entry[split].update(classification_stats(s, y, rate))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--project', default='ant')
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46])
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--patience', type=int, default=15, help='early stopping sur la validation (0 = désactivé)')
    ap.add_argument('--min-epochs', type=int, default=10,
                    help='aucune sélection de checkpoint / arrêt anticipé avant cette époque')
    ap.add_argument('--variants', nargs='+', default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument('--tune', action='store_true', help='grille lambda_c / lambda_r / température sur la validation')
    ap.add_argument('--improved', action='store_true',
                    help='réentraînement train+validation, pos_weight, compression log des métriques')
    ap.add_argument('--out', default=str(ROOT / 'results'))
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    torch.set_num_threads(min(4, torch.get_num_threads()))
    rel = releases_of(args.project)
    tr, val, te = rel[0], (rel[1] if len(rel) >= 3 else None), rel[-1]
    data = {v: load(ROOT / 'data' / args.project / f'{args.project}-{v}.npz', args.device) for v in rel}
    if args.improved:
        for d in data.values():
            d['x'] = signed_log(d['x'])
    refit = args.improved and val is not None
    out_dir = Path(args.out) / (args.project + ('_improved' if args.improved else ''))
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Projet {args.project} | train={tr} validation={val} test={te} | device={args.device} | '
          f'improved={args.improved} (refit train+val={refit})')
    for v in rel:
        d = data[v]
        print(f'  {v}: X{tuple(d["x"].shape)} edges{tuple(d["edges"].shape)} positifs={int(d["y"].sum())}/{len(d["y"])}')

    tuned = {}
    if args.tune:
        if val is None:
            print('!! --tune ignoré : pas de version de validation (2 fenêtres). Hyperparamètres par défaut.')
        else:
            tuned = tune(data, tr, val, args.epochs, args.patience, args.device, args.seeds[0],
                         args.variants, args.min_epochs, args.improved)
            (out_dir / 'tuning.json').write_text(json.dumps(tuned, indent=2), encoding='utf-8')

    variants = [v for v in args.variants if v != 'A8_ensemble']
    if 'A8_ensemble' in args.variants:
        missing = [c for c in ENSEMBLE_OF if c not in variants]
        if missing:
            print(f'!! A8_ensemble ignoré : composantes absentes {missing}')
        else:
            variants.append('A8_ensemble')

    rows = []
    for seed in args.seeds:
        sdir = out_dir / f'seed{seed}'
        sdir.mkdir(exist_ok=True)
        for f in sdir.glob('*_scores.csv'):
            f.unlink()
        result = {'project': args.project, 'train': tr, 'validation': val, 'test': te, 'seed': seed,
                  'epochs': args.epochs, 'min_epochs': args.min_epochs, 'patience': args.patience,
                  'improved': args.improved, 'refit_train_plus_validation': refit,
                  'caveat': 'Current-version defect detection, NOT next-version forecasting', 'variants': {}}
        store = {}   # scores par variante et par split (pour l'ensemble)
        for name in variants:
            set_seed(seed)
            t0 = time.time()
            cfg = VARIANTS[name]
            ytr = data[tr]['y'].cpu().numpy()
            entry = {}
            sc = {}   # split -> scores ; 'validation' = modèle de SÉLECTION (jamais entraîné sur la validation)
            outs = {}
            if cfg == 'ensemble':
                for split in store[ENSEMBLE_OF[0]]:
                    sc[split] = np.mean([rank01(store[c][split]) for c in ENSEMBLE_OF], axis=0)
                entry['components'] = ENSEMBLE_OF
            elif cfg is None:
                def fit_hgb(keys):
                    m = HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=7, learning_rate=.05, random_state=seed)
                    m.fit(np.concatenate([data[k]['x'][:, -1].cpu().numpy() for k in keys]),
                          np.concatenate([data[k]['y'].cpu().numpy() for k in keys]))
                    return m
                sel = fit_hgb([tr])
                final = fit_hgb([tr, val]) if refit else sel
                if val is not None:
                    sc['validation'] = sel.predict_proba(data[val]['x'][:, -1].cpu().numpy())[:, 1]
                sc['test'] = final.predict_proba(data[te]['x'][:, -1].cpu().numpy())[:, 1]
                entry['parameters'] = None
            else:
                p = tuned.get(name, {}).get('params') or {}
                c = {**cfg, **{k: v for k, v in p.items() if k.startswith('lambda')}}
                hp = {**DEFAULT_HP, **{k: v for k, v in p.items() if k == 'temperature'}}
                net, best_ep, _ = train_variant(c, hp, data, [tr], val, args.epochs, args.patience, args.device,
                                                args.min_epochs, args.improved)
                if val is not None:
                    with torch.no_grad():
                        outs['validation'] = net(data[val]['x'], data[val]['edges'])
                    sc['validation'] = score_of(outs['validation'], c).cpu().numpy()
                if refit:   # même nombre d'époques que celui choisi sur la validation, données train + validation
                    set_seed(seed)
                    net, _, _ = train_variant(c, hp, data, [tr, val], None, best_ep, 0, args.device,
                                              args.min_epochs, args.improved)
                with torch.no_grad():
                    outs['test'] = net(data[te]['x'], data[te]['edges'])
                sc['test'] = score_of(outs['test'], c).cpu().numpy()
                entry.update({'selected_epoch': best_ep, 'parameters': sum(x.numel() for x in net.parameters()),
                              'lambda_c': c['lambda_c'], 'lambda_r': c['lambda_r'], 'temperature': hp['temperature']})
                torch.save({'state_dict': net.state_dict(), 'config': c, 'hp': hp, 'improved': args.improved,
                            'n_features': data[tr]['x'].shape[-1]}, sdir / f'{name}.pt')
            store[name] = sc
            # Règle de décision : fraction de classes signalées réglée sur la validation (sinon : prévalence du train)
            rate = choose_flag_rate(sc['validation'], data[val]['y'].cpu().numpy()) if 'validation' in sc \
                else float(ytr.mean())
            entry['decision_flag_rate'] = round(rate, 4)
            for split, v in (('validation', val), ('test', te)):
                if split not in sc:
                    continue
                evaluate(entry, split, sc[split], data[v], rate)
                if split in outs:
                    entry[split]['mean_alpha_temporal'] = round(float(outs[split]['gate'][:, 0].mean()), 4)
                    entry[split]['mean_agreement'] = round(float(outs[split]['agreement'].mean()), 4)
                export_scores(sdir / f'{name}_scores.csv', split, data[v], sc[split], outs.get(split))
            entry['training_seconds'] = round(time.time() - t0, 2)
            result['variants'][name] = entry
            t = entry['test']
            print(f'[seed {seed}] {name:30s} AP={t["AP"]:.4f} Acc={t["Accuracy"]:.4f} (majorité {t["Majority_accuracy"]:.3f}) '
                  f'BalAcc={t["Balanced_accuracy"]:.4f} MCC={t["MCC"]:.4f} R@20%LOC(risque/LOC)={t["EA_Recall@20pct_LOC"]:.4f} '
                  f'({entry["training_seconds"]}s)', flush=True)
            rows.append({'variant': name, 'seed': seed, **{f'test_{k}': v for k, v in entry['test'].items()},
                         **({f'val_{k}': v for k, v in entry['validation'].items()} if 'validation' in entry else {}),
                         'training_seconds': entry['training_seconds']})
        (sdir / 'metrics.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')

    # Résumé moyenne ± écart-type
    metrics = ['test_AP', 'test_ROC_AUC', 'test_Accuracy', 'test_Majority_accuracy', 'test_Balanced_accuracy',
               'test_MCC', 'test_F1', 'test_Recall@20pct_classes', 'test_Recall@20pct_LOC',
               'test_EA_Recall@20pct_LOC', 'val_AP', 'val_Accuracy', 'val_EA_Recall@20pct_LOC', 'training_seconds']
    summary = []
    for name in variants:
        r = [x for x in rows if x['variant'] == name]
        line = {'variant': name, 'n_seeds': len(r)}
        for m in metrics:
            vals = [x[m] for x in r if x.get(m) is not None]
            if vals:
                line[f'{m}_mean'] = round(float(np.mean(vals)), 4)
                line[f'{m}_std'] = round(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0., 4)
        summary.append(line)
    keys = sorted({k for s in summary for k in s}, key=lambda k: (k != 'variant', k != 'n_seeds', k))
    with (out_dir / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(summary)
    print(f'\n=== Résumé test ({len(args.seeds)} graines, improved={args.improved}) ===')
    for s in summary:
        print(f"{s['variant']:30s} AP={s.get('test_AP_mean', 0):.4f}±{s.get('test_AP_std', 0):.4f}  "
              f"Acc={s.get('test_Accuracy_mean', 0):.4f}±{s.get('test_Accuracy_std', 0):.4f} "
              f"(majorité {s.get('test_Majority_accuracy_mean', 0):.3f})  "
              f"MCC={s.get('test_MCC_mean', 0):.4f}  "
              f"R@20%LOC(risque/LOC)={s.get('test_EA_Recall@20pct_LOC_mean', 0):.4f}")
    print('Fichiers :', out_dir)


if __name__ == '__main__':
    main()
