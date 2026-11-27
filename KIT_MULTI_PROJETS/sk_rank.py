"""PRIORITEST-STAR — step 6.2 of ARCHITECTURE.md: Scott-Knott ESD ranking of all models, per project and metric.

Inputs are the saved test scores of the fair protocols (each model tuned on its project's own validation data):
  Ant   → results/phase1/ant/test_scores.npz     (tuned on Ant 1.6)
  Camel → results/retune/camel_test_scores.npz   (tuned by 3-fold split of Camel 1.4)
  POI   → results/retune/poi_test_scores.npz     (tuned by 3-fold split of POI 2.5)
For each metric, every model is scored on the SAME 1 000 class-bootstrap resamples of the test release, then
Scott-Knott ESD groups models whose difference is not significant or negligible (rank 1 = best group).
AP uses the rank ensembles; Popt (score / LOC order) uses the mean-score ensembles.
Usage: python sk_rank.py
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
import stats as S  # noqa: E402

MODELS = ['A1_HGB', 'A2_TCN_BCE', 'A3_graph_classic_fusion_BCE', 'A4_similarity_fusion_BCE', 'A5_plus_InfoNCE',
          'A6_STAR_full', 'B_LR', 'B_RF', 'B_ET', 'STAR_R_blend']
SOURCES = {'ant': 'phase1/ant/test_scores.npz', 'camel': 'retune/camel_test_scores.npz', 'poi': 'retune/poi_test_scores.npz'}


def main():
    res = {}
    for P, rel in SOURCES.items():
        f = ROOT / 'results' / rel
        if not f.exists():
            print(f'skip {P}: {f} missing'); continue
        z = np.load(f); y = z['y']; loc = z['loc']; res[P] = {}
        for metric, prefix in [('AP', ''), ('Popt_density', 'prob_')]:
            scores = {m: z[prefix + m] for m in MODELS if prefix + m in z.files}
            samp = S.bootstrap_samples(scores, y, loc, metric, 1000)
            rank = S.scott_knott_esd(samp)
            res[P][metric] = {m: {'rank': rank[m], 'mean': round(float(samp[m].mean()), 4)} for m in scores}
            print(f'\n== {P} {metric}')
            for m in sorted(rank, key=lambda k: (rank[k], -samp[k].mean())):
                print(f'  rank {rank[m]}  {m:30s} {samp[m].mean():.4f}')
    (ROOT / 'results' / 'scott_knott.json').write_text(json.dumps(res, indent=1))


if __name__ == '__main__':
    main()
