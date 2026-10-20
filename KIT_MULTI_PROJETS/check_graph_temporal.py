from pathlib import Path
import numpy as np, json
root=Path(__file__).resolve().parent
m=json.loads((root/'data/graph_temporal_manifest.json').read_text()); count=0
for row in m['windows']:
 project=row['project']; version=row['target']; path=root/'data'/project/f'{project}-{version}_graph_temporal.npz'; d=np.load(path)
 old=np.load(root/'data'/project/f'{project}-{version}.npz')
 assert len(d['versions'])==3
 assert d['temporal_links'].shape[1]==4
 assert d['spatial_edges_global'].shape[1]==2
 assert d['temporal_edges_global'].shape[1]==2
 for t in range(3):
  ids=d[f'ids_v{t}'];e=d[f'edges_v{t}'];feat=d[f'features_raw_v{t}']
  assert len(ids)==len(feat) and feat.shape[1]==20
  assert np.isfinite(feat).all()
  assert e.shape[1]==2
  if len(e):assert np.min(e)>=0 and np.max(e)<len(ids)
 for t,i,t2,j in d['temporal_links']:
  assert t2==t+1 and d[f'ids_v{t}'][i]==d[f'ids_v{t2}'][j]
 for i,index in enumerate(d['target_indices_last']):assert old['ids'][i]==d['ids_v2'][index]
 assert len(d['temporal_edges_global'])==len(d['temporal_links'])
 count+=1
print(f'OK : {count} fenêtres, {len(set(x["project"] for x in m["windows"]))} projets')
