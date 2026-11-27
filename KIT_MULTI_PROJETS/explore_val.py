"""Exploration des hyperparamètres sur la VALIDATION uniquement (le test n'est jamais chargé).

Pour chaque configuration : entraînement sur la fenêtre train, courbe d'AP de validation par époque pour
plusieurs graines, puis époque choisie sur la courbe MOYENNE (lissée) — plus robuste que l'argmax par graine.
Usage : python explore_val.py --project ant --out results/explore_ant.json
"""
import argparse, itertools, json, sys, time
from pathlib import Path
import numpy as np, torch
from sklearn.metrics import average_precision_score
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from run_star import VARIANTS, build_net, cohort_loss, score_of, set_seed, signed_log, releases_of  # noqa
from train import load  # noqa


def curve(cfg, hp, data, tr, val, epochs, seed, device, pos_weight):
    set_seed(seed)
    net = build_net(cfg, data[tr]['x'].shape[-1], device)
    opt = torch.optim.Adam(net.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay'])
    yv = data[val]['y'].cpu().numpy(); aps = []
    for ep in range(epochs):
        net.train(); opt.zero_grad()
        loss = cohort_loss(net, cfg, hp, data[tr], pos_weight); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
        net.eval()
        with torch.no_grad():
            aps.append(average_precision_score(yv, score_of(net(data[val]['x'], data[val]['edges']), cfg).cpu().numpy()))
    return np.array(aps)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--project', default='ant'); p.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    p.add_argument('--epochs', type=int, default=150); p.add_argument('--out', default=str(ROOT / 'results' / 'explore_ant.json'))
    a = p.parse_args(); torch.set_num_threads(2)
    rel = releases_of(a.project); assert len(rel) >= 3, 'validation release required'
    tr, val = rel[0], rel[1]                       # le test (rel[2]) n'est PAS chargé
    data = {v: load(ROOT / 'data' / a.project / f'{a.project}-{v}.npz', 'cpu') for v in (tr, val)}
    for d in data.values(): d['x'] = signed_log(d['x'])
    base = VARIANTS['A6_STAR_full']
    grid = dict(lr=[1e-3, 3e-3], weight_decay=[1e-4, 1e-3], dropout=[0., .2], lambda_c=[0., .1, .25],
                lambda_r=[.25, 1.], bidirectional=[False, True], use_delta=[False, True])
    rows = []; t0 = time.time()
    for combo in itertools.product(*grid.values()):
        g = dict(zip(grid, combo))
        cfg = {**base, 'dropout': g['dropout'], 'lambda_c': g['lambda_c'], 'lambda_r': g['lambda_r'],
               'bidirectional': g['bidirectional'], 'use_delta': g['use_delta']}
        hp = dict(lr=g['lr'], weight_decay=g['weight_decay'], temperature=.2)
        C = np.stack([curve(cfg, hp, data, tr, val, a.epochs, s, 'cpu', True) for s in a.seeds])
        m = np.convolve(C.mean(0), np.ones(5) / 5, mode='same')     # courbe moyenne lissée
        m[:10] = -1                                                  # plancher min_epochs
        ep = int(np.argmax(m)) + 1
        rows.append({**g, 'best_epoch': ep, 'val_AP_at_epoch': round(float(C.mean(0)[ep - 1]), 4),
                     'val_AP_smoothed': round(float(m[ep - 1]), 4), 'val_AP_seed_std': round(float(C[:, ep - 1].std()), 4)})
    rows.sort(key=lambda r: -r['val_AP_smoothed'])
    Path(a.out).write_text(json.dumps({'project': a.project, 'train': tr, 'validation': val, 'seeds': a.seeds,
                                       'seconds': round(time.time() - t0, 1), 'results': rows}, indent=1))
    print(f'{len(rows)} configs in {time.time()-t0:.0f}s')
    for r in rows[:12]: print(r)


if __name__ == '__main__':
    main()
