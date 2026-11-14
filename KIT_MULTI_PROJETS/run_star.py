"""PRIORITEST-STAR — lanceur complet des ablations (fiche PFA §6) + variante améliorée STAR+.

Exemples :
  python run_star.py --project ant --seeds 42 43 44
  python run_star.py --project jedit --seeds 42 43 44 45 46 --tune
  python run_star.py --project ant --variants A2_TCN_BCE A6_STAR_full --epochs 20

Protocole :
  3 fenêtres cibles : 1re = entraînement, 2e = validation (early stopping + réglages), 3e = test final.
  2 fenêtres       : 1re = entraînement, 2e = test exploratoire ; nombre d'époques fixe, aucun réglage.
  Détection des défauts de la version observée — PAS une prédiction de la version suivante.

Sorties (results/<projet>/) :
  seed<k>/metrics.json, seed<k>/<variante>_scores.csv  (scores, alpha, beta, accord cosinus par classe)
  summary.csv  (moyenne ± écart-type sur les graines)   tuning.json (si --tune)
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
from model import (CrossViewSTAR, cross_view_infonce, label_aware_infonce,  # noqa: E402
                   pairwise_rank_loss, effort_aware_rank_loss)
from train import load, stats, effort_aware_stats  # noqa: E402

# Chaque ligne ajoute UN mécanisme à la précédente (A1..A6), A7 = améliorations proposées.
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
}
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


def check_finite(out, where):
    for k, v in out.items():
        if torch.is_tensor(v) and not torch.isfinite(v).all():
            raise FloatingPointError(f"NaN/Inf dans '{k}' ({where})")


def score_of(out, cfg):
    return torch.sigmoid(out['rank'] if cfg['lambda_r'] else out['logits'])


def train_variant(cfg, hp, data, tr, val, epochs, patience, device, min_epochs=10):
    """Entraîne une variante neuronale (batch complet = toutes les classes de la version d'entraînement).

    min_epochs : aucune sélection de checkpoint ni arrêt anticipé avant cette époque. Sans ce plancher, une AP de
    validation bruitée peut retenir un modèle quasi non entraîné (époque 1), ce qui rend les ablations instables.
    """
    min_epochs = max(1, min(min_epochs, epochs))
    net = CrossViewSTAR(data[tr]['x'].shape[-1], use_graph=cfg['use_graph'], similarity_gate=cfg['similarity_gate'],
                        use_delta=cfg.get('use_delta', False), bidirectional=cfg.get('bidirectional', False),
                        dropout=cfg.get('dropout', 0.)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay'])
    a = data[tr]
    loc_t = torch.as_tensor(a['loc'], device=device, dtype=torch.float32)
    best, best_state, best_ep, wait = -1., None, epochs, 0
    for ep in range(1, epochs + 1):
        net.train(); opt.zero_grad()
        out = net(a['x'], a['edges'])
        check_finite(out, f'epoch {ep}')
        loss = F.binary_cross_entropy_with_logits(out['logits'], a['y'])
        if cfg['lambda_r']:
            rl = effort_aware_rank_loss(out['rank'], a['y'], loc_t) if cfg.get('effort_aware') \
                else pairwise_rank_loss(out['rank'], a['y'])
            loss = loss + cfg['lambda_r'] * rl
        if cfg['lambda_c']:
            cl = label_aware_infonce(out['temporal'], out['structural'], a['y'], hp['temperature']) if cfg.get('label_aware') \
                else cross_view_infonce(out['temporal'], out['structural'], hp['temperature'])
            loss = loss + cfg['lambda_c'] * cl
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


def tune(project, data, tr, val, epochs, patience, device, seed, names, min_epochs=10):
    """Recherche en grille sur la VALIDATION uniquement (jamais sur le test)."""
    chosen = {}
    for name in names:
        cfg = VARIANTS[name]
        if cfg is None or (not cfg['lambda_c'] and not cfg['lambda_r']):
            continue
        keys = [k for k in GRID if (k != 'lambda_r' or cfg['lambda_r']) and (k == 'lambda_r' or cfg['lambda_c'])]
        best = (-1, None)
        for combo in itertools.product(*[GRID[k] for k in keys]):
            p = dict(zip(keys, combo))
            c = {**cfg, **{k: v for k, v in p.items() if k.startswith('lambda')}}
            hp = {**DEFAULT_HP, **{k: v for k, v in p.items() if k == 'temperature'}}
            set_seed(seed)
            _, _, ap = train_variant(c, hp, data, tr, val, epochs, patience, device, min_epochs)
            print(f'  [tune] {name} {p} -> AP_val={ap:.4f}', flush=True)
            if ap > best[0]:
                best = (ap, p)
        chosen[name] = {'params': best[1], 'AP_validation': round(best[0], 4)}
        print(f'[tune] {name} retenu : {best[1]} (AP_val={best[0]:.4f})', flush=True)
    return chosen


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--project', default='ant')
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--patience', type=int, default=15, help='early stopping sur la validation (0 = désactivé)')
    ap.add_argument('--min-epochs', type=int, default=10,
                    help='aucune sélection de checkpoint / arrêt anticipé avant cette époque')
    ap.add_argument('--variants', nargs='+', default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument('--tune', action='store_true', help='grille lambda_c / lambda_r / température sur la validation')
    ap.add_argument('--out', default=str(ROOT / 'results'))
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    torch.set_num_threads(min(4, torch.get_num_threads()))
    rel = releases_of(args.project)
    tr, val, te = rel[0], (rel[1] if len(rel) >= 3 else None), rel[-1]
    data = {v: load(ROOT / 'data' / args.project / f'{args.project}-{v}.npz', args.device) for v in rel}
    out_dir = Path(args.out) / args.project
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f'Projet {args.project} | train={tr} validation={val} test={te} | device={args.device}')
    for v in rel:
        d = data[v]
        print(f'  {v}: X{tuple(d["x"].shape)} edges{tuple(d["edges"].shape)} positifs={int(d["y"].sum())}/{len(d["y"])}')

    tuned = {}
    if args.tune:
        if val is None:
            print('!! --tune ignoré : pas de version de validation (2 fenêtres). Hyperparamètres par défaut.')
        else:
            tuned = tune(args.project, data, tr, val, args.epochs, args.patience, args.device, args.seeds[0],
                         args.variants, args.min_epochs)
            (out_dir / 'tuning.json').write_text(json.dumps(tuned, indent=2), encoding='utf-8')

    rows = []
    for seed in args.seeds:
        sdir = out_dir / f'seed{seed}'
        sdir.mkdir(exist_ok=True)
        for f in sdir.glob('*_scores.csv'):
            f.unlink()
        result = {'project': args.project, 'train': tr, 'validation': val, 'test': te, 'seed': seed,
                  'epochs': args.epochs, 'min_epochs': args.min_epochs, 'patience': args.patience,
                  'caveat': 'Current-version defect detection, NOT next-version forecasting', 'variants': {}}
        for name in args.variants:
            set_seed(seed)
            t0 = time.time()
            cfg = VARIANTS[name]
            splits = [(s, v) for s, v in (('validation', val), ('test', te)) if v is not None]
            if cfg is None:
                hgb = HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=7, learning_rate=.05, random_state=seed)
                hgb.fit(data[tr]['x'][:, -1].cpu().numpy(), data[tr]['y'].cpu().numpy())
                entry = {'parameters': None}
                for split, v in splits:
                    s = hgb.predict_proba(data[v]['x'][:, -1].cpu().numpy())[:, 1]
                    entry[split] = stats(s, data[v]['y'].cpu().numpy(), data[v]['loc'])
                    entry[split].update(effort_aware_stats(s, data[v]['y'].cpu().numpy(), data[v]['loc']))
                    export_scores(sdir / f'{name}_scores.csv', split, data[v], s)
            else:
                p = tuned.get(name, {}).get('params') or {}
                c = {**cfg, **{k: v for k, v in p.items() if k.startswith('lambda')}}
                hp = {**DEFAULT_HP, **{k: v for k, v in p.items() if k == 'temperature'}}
                net, best_ep, _ = train_variant(c, hp, data, tr, val, args.epochs, args.patience, args.device,
                                                args.min_epochs)
                entry = {'selected_epoch': best_ep, 'parameters': sum(x.numel() for x in net.parameters()),
                         'lambda_c': c['lambda_c'], 'lambda_r': c['lambda_r'], 'temperature': hp['temperature']}
                for split, v in splits:
                    with torch.no_grad():
                        o = net(data[v]['x'], data[v]['edges'])
                        s = score_of(o, c).cpu().numpy()
                    entry[split] = stats(s, data[v]['y'].cpu().numpy(), data[v]['loc'])
                    entry[split].update(effort_aware_stats(s, data[v]['y'].cpu().numpy(), data[v]['loc']))
                    entry[split]['mean_alpha_temporal'] = round(float(o['gate'][:, 0].mean()), 4)
                    entry[split]['mean_agreement'] = round(float(o['agreement'].mean()), 4)
                    export_scores(sdir / f'{name}_scores.csv', split, data[v], s, o)
                torch.save({'state_dict': net.state_dict(), 'config': c, 'hp': hp,
                            'n_features': data[tr]['x'].shape[-1]}, sdir / f'{name}.pt')
            entry['training_seconds'] = round(time.time() - t0, 2)
            result['variants'][name] = entry
            print(f'[seed {seed}] {name:30s} test AP={entry["test"]["AP"]:.4f} '
                  f'R@20%LOC={entry["test"]["Recall@20pct_LOC"]:.4f} '
                  f'R@20%LOC(risque/LOC)={entry["test"]["EA_Recall@20pct_LOC"]:.4f} ({entry["training_seconds"]}s)', flush=True)
            rows.append({'variant': name, 'seed': seed, **{f'test_{k}': v for k, v in entry['test'].items()},
                         **({f'val_{k}': v for k, v in entry['validation'].items()} if 'validation' in entry else {}),
                         'training_seconds': entry['training_seconds']})
        (sdir / 'metrics.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')

    # Résumé moyenne ± écart-type
    metrics = ['test_AP', 'test_ROC_AUC', 'test_Recall@20pct_classes', 'test_Recall@20pct_LOC',
               'test_EA_Recall@20pct_LOC', 'val_AP', 'val_EA_Recall@20pct_LOC', 'training_seconds']
    summary = []
    for name in args.variants:
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
    print('\n=== Résumé (test, moyenne ± écart-type sur', len(args.seeds), 'graines) ===')
    for s in summary:
        print(f"{s['variant']:30s} AP={s.get('test_AP_mean', 0):.4f}±{s.get('test_AP_std', 0):.4f}  "
              f"R@20%LOC={s.get('test_Recall@20pct_LOC_mean', 0):.4f}±{s.get('test_Recall@20pct_LOC_std', 0):.4f}  "
              f"R@20%LOC(risque/LOC)={s.get('test_EA_Recall@20pct_LOC_mean', 0):.4f}"
              f"±{s.get('test_EA_Recall@20pct_LOC_std', 0):.4f}")
    print('Fichiers :', out_dir)


if __name__ == '__main__':
    main()
