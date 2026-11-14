import argparse,json,time,random,copy
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score,roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
from model import CrossViewSTAR,cross_view_infonce,pairwise_rank_loss

def load(path,device):
    f=np.load(path,allow_pickle=False)
    return {'x':torch.as_tensor(f['X'],device=device),'y':torch.as_tensor(f['y'],device=device),
            'edges':torch.as_tensor(f['edges'],device=device),'loc':f['loc'],'ids':f['ids']}

def stats(scores,y,loc):
    s=np.asarray(scores);y=np.asarray(y).astype(int);loc=np.maximum(np.asarray(loc),1)
    ix=np.argsort(-s,kind='stable');budget=.2*loc.sum();used=0;chosen=[]
    for k in ix:
        if used+loc[k]>budget:continue
        used+=loc[k];chosen.append(k)
    return {'AP':round(float(average_precision_score(y,s)),4),'ROC_AUC':round(float(roc_auc_score(y,s)),4) if len(np.unique(y))>1 else None,
            'Recall@20pct_classes':round(float(y[ix[:max(1,int(np.ceil(.2*len(y))))]].sum()/max(1,y.sum())),4),
            'Recall@20pct_LOC':round(float(y[chosen].sum()/max(1,y.sum())),4),'classes_in_budget':len(chosen)}

def run(data='data',out='results',epochs=60,seed=42,device='cpu'):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.set_num_threads(min(4,torch.get_num_threads()))
    output=Path(out);output.mkdir(parents=True,exist_ok=True)
    data=Path(data)
    if not (data/'ant-1.5.npz').exists() and (data/'ant'/'ant-1.5.npz').exists():
        data=data/'ant'  # organisation multi-projets : data/<projet>/<projet>-<version>.npz
    cohorts={v:load(data/f'ant-{v}.npz',device) for v in ('1.5','1.6','1.7')}
    result={'protocol':'ANT within-target-version defect detection; NOT next-version prospective forecast', 'seed':seed,'epochs':epochs,'models':{}}
    baseline=HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=7,learning_rate=.05,random_state=seed)
    baseline.fit(cohorts['1.5']['x'][:,-1].cpu().numpy(),cohorts['1.5']['y'].cpu().numpy())
    result['baseline_HGB']={v:stats(baseline.predict_proba(cohorts[v]['x'][:,-1].cpu().numpy())[:,1],cohorts[v]['y'].cpu().numpy(),cohorts[v]['loc']) for v in ['1.6','1.7']}
    variants={'classic_gate_BCE_Rank':(False,0.,.25),'similarity_gate_BCE_Rank':(True,0.,.25),
              'similarity_gate_contrastive':(True,.25,0.),'STAR_CrossView_full':(True,.25,.25)}
    for name,(sim_gate,lambda_c,lambda_r) in variants.items():
        net=CrossViewSTAR(cohorts['1.5']['x'].shape[-1],similarity_gate=sim_gate).to(device)
        optim=torch.optim.Adam(net.parameters(),lr=.002,weight_decay=1e-4)
        best=-1.;best_state=None;best_epoch=0;start=time.time()
        for ep in range(1,epochs+1):
            net.train();optim.zero_grad();a=cohorts['1.5'];p=net(a['x'],a['edges'])
            loss=F.binary_cross_entropy_with_logits(p['logits'],a['y'])+lambda_r*pairwise_rank_loss(p['rank'],a['y'])
            if lambda_c:loss=loss+lambda_c*cross_view_infonce(p['temporal'],p['structural'])
            loss.backward();optim.step()
            net.eval()
            with torch.no_grad():
                b=cohorts['1.6'];pv=net(b['x'],b['edges']);vscore=torch.sigmoid(pv['rank'] if lambda_r else pv['logits']).cpu().numpy()
            ap=average_precision_score(b['y'].cpu().numpy(),vscore)
            if ap>best:best=ap;best_state=copy.deepcopy({k:v.detach().cpu() for k,v in net.state_dict().items()});best_epoch=ep
        net.load_state_dict(best_state);net.eval();entry={'best_epoch':best_epoch,'training_seconds':round(time.time()-start,2),
              'parameters':sum(x.numel() for x in net.parameters()),'lambda_c':lambda_c,'lambda_r':lambda_r}
        for v in ['1.6','1.7']:
            with torch.no_grad():
                a=cohorts[v];p=net(a['x'],a['edges']);s=torch.sigmoid(p['rank'] if lambda_r else p['logits']).cpu().numpy()
            entry[v]=stats(s,a['y'].cpu().numpy(),a['loc'])
        result['models'][name]=entry
        torch.save({'state_dict':best_state,'n_features':cohorts['1.5']['x'].shape[-1],
                    'similarity_gate':sim_gate,'lambda_c':lambda_c,'lambda_r':lambda_r},output/f'{name}.pt')
        print(name,entry,flush=True)
    (output/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));return result

if __name__=='__main__':
    ROOT=Path(__file__).resolve().parent.parent
    p=argparse.ArgumentParser();p.add_argument('--data',default=str(ROOT/'data'));p.add_argument('--out',default=str(ROOT/'results'));p.add_argument('--epochs',type=int,default=60);p.add_argument('--seed',type=int,default=42);p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');args=p.parse_args()
    run(args.data,args.out,args.epochs,args.seed,args.device)
