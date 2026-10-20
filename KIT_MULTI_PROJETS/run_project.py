"""Exécute un pilote reproductible sur un projet avec au moins 2 fenêtres cibles.
3+ fenêtres : première entraînement, deuxième validation, dernière test.
2 fenêtres : première entraînement, dernière test exploratoire; aucun réglage sur le test.
1 fenêtre : seulement descriptif (aucun holdout chronologique possible).
"""
import argparse,sys,json,random,copy
from pathlib import Path
import numpy as np,torch,torch.nn.functional as F
from sklearn.metrics import average_precision_score
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
from train import load,stats
from model import CrossViewSTAR,cross_view_infonce,pairwise_rank_loss

def run(project,epochs=35,seed=42,device='cpu'):
 m=json.loads((ROOT/'data'/'manifest_multi.json').read_text());info=m['projects'].get(project)
 if not info:raise ValueError(f"Projet indisponible: {project}. Voir manifest_multi.json")
 releases=[x['version'] for x in info['targets'] if x['status']=='prepared']
 if len(releases)<2:raise ValueError(f"{project}: {len(releases)} seule fenêtre; comparaison temporelle train/test impossible. Utiliser comme jeu descriptif ou dans une étude inter-projets explicite.")
 torch.set_num_threads(min(4,torch.get_num_threads()));random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 cs={v:load(ROOT/'data'/project/f'{project}-{v}.npz',device) for v in releases}
 tr,val,te=releases[0],(releases[1] if len(releases)>=3 else None),releases[-1]
 result={'project':project,'train':tr,'validation':val,'test':te,'seed':seed,'epochs':epochs,'caveat':'Current-version detection, not prospective next-version forecasting','variants':{}}
 variants={'classic_gate_BCE_Rank':(False,0.,.25),'similarity_gate_BCE_Rank':(True,0.,.25),'similarity_gate_contrastive':(True,.25,0.),'STAR_CrossView_full':(True,.25,.25)}
 for name,(sim,lamc,lamr) in variants.items():
  model=CrossViewSTAR(cs[tr]['x'].shape[-1],similarity_gate=sim).to(device)
  optimizer=torch.optim.Adam(model.parameters(),lr=.002,weight_decay=1e-4)
  best=-1;state=None;chosen=epochs
  for ep in range(1,epochs+1):
   model.train();optimizer.zero_grad();a=cs[tr];pred=model(a['x'],a['edges'])
   loss=F.binary_cross_entropy_with_logits(pred['logits'],a['y'])+lamr*pairwise_rank_loss(pred['rank'],a['y'])
   if lamc:loss=loss+lamc*cross_view_infonce(pred['temporal'],pred['structural'])
   if not torch.isfinite(loss):raise ValueError('nonfinite loss')
   loss.backward();optimizer.step()
   if val is not None:
    model.eval()
    with torch.no_grad():
     b=cs[val];o=model(b['x'],b['edges']);scores=torch.sigmoid(o['rank'] if lamr else o['logits']).cpu().numpy()
    ap=average_precision_score(b['y'].cpu().numpy(),scores)
    if ap>best:best=ap;chosen=ep;state=copy.deepcopy(model.state_dict())
  if state is not None:model.load_state_dict(state)
  model.eval()
  with torch.no_grad():
   b=cs[te];pred=model(b['x'],b['edges']);scores=torch.sigmoid(pred['rank'] if lamr else pred['logits']).cpu().numpy()
  result['variants'][name]={'selected_epoch':chosen,'test':stats(scores,b['y'].cpu().numpy(),b['loc'])}
  print(project,name,result['variants'][name],flush=True)
 out=ROOT/'results';out.mkdir(exist_ok=True)
 (out/f'{project}_seed{seed}.json').write_text(json.dumps(result,indent=2))
 return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--project',default='jedit');p.add_argument('--epochs',type=int,default=35);p.add_argument('--seed',type=int,default=42);p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');a=p.parse_args();run(a.project,a.epochs,a.seed,a.device)
