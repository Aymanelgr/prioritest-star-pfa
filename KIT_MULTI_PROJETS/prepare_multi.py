"""Prépare des fenêtres STRICTES de trois versions PROMISE, sans données futures.
Usage: python prepare_multi.py --archive PROMISE-dataset-main.zip --out data
"""
import argparse,collections,io,json,re,zipfile
from pathlib import Path
import numpy as np,pandas as pd
ROOT='PROMISE-dataset-main/PROMISE/'
FEATURES=['wmc','dit','noc','cbo','rfc','lcom','ca','ce','npm','lcom3','loc','dam','moa','mfa','cam','ic','cbm','amc','max_cc','avg_cc']

def numeric_order(s):
    if s=='init':return (-1,)
    return tuple(int(x) for x in re.findall(r'\d+',s))

def main(archive,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        entries=z.namelist(); grouped=collections.defaultdict(dict); source_files=collections.defaultdict(list)
        for p in entries:
            m=re.match(re.escape(ROOT)+r'labeled-data/([^/]+)/\1-(.+)\.csv$',p)
            if m:grouped[m.group(1)][m.group(2)]=p
            m=re.match(re.escape(ROOT)+r'source-code/([^/]+)/([^/]+)/.*\.java$',p)
            if m:source_files[(m.group(1),m.group(2))].append(p)
        manifest={'source_archive':'PROMISE-dataset-main.zip','window_size':3,'features':FEATURES,'projects':{},'excluded':{},'protocol':'Current-version defect detection. Temporal prospective forecasting NOT validated.','graph':'Only explicit non-static Java imports (proxy, not complete dependency graph).','normalization':'Train-only fit for each project; used for later release windows.','labels':'bug>0 binary; target release itself is present in features; no future version data used'}
        for project,verpaths in sorted(grouped.items()):
            versions=sorted(verpaths,key=numeric_order)
            if len(versions)<3:
                manifest['excluded'][project]={'versions':versions,'reason':'fewer than three releases'};continue
            frames={};sources={};rowcounts={}
            for v in versions:
                df=pd.read_csv(io.BytesIO(z.read(verpaths[v])))
                if not set(FEATURES+['name1','bug']).issubset(df.columns):raise ValueError((project,v,df.columns))
                df=df.dropna(subset=['name1']).drop_duplicates('name1').set_index('name1')
                frames[v]=df;rowcounts[v]=len(df)
                # match source via qualified name -> pathname suffix (no Java parsing for this step).
                name_lookup=collections.defaultdict(list)
                for filepath in source_files.get((project,f'{project}-{v}'),[]):
                    name_lookup[filepath.rsplit('/',1)[-1][:-5]].append(filepath)
                usable={}
                for fqcn in df.index:
                    simple=fqcn.rsplit('.',1)[-1];suffix=fqcn.replace('.','/')+'.java'
                    cands=[p for p in name_lookup.get(simple,[]) if p.endswith('/'+suffix)]
                    if len(cands)==1:usable[fqcn]=cands[0]
                sources[v]=usable
            target_stats=[]; raw_data={}
            for vi in range(2,len(versions)):
                v=versions[vi];window=versions[vi-2:vi+1]
                keys=sorted(set.intersection(*(set(frames[k].index) for k in window)))
                if not keys:
                    target_stats.append({'version':v,'window':window,'n':0,'status':'no overlapping classes'});continue
                raw=np.stack([frames[w].loc[keys,FEATURES].apply(pd.to_numeric,errors='coerce').to_numpy(dtype='float32') for w in window],axis=1)
                bugs=pd.to_numeric(frames[v].loc[keys,'bug'],errors='coerce').to_numpy()
                good=np.isfinite(bugs);keys=[k for k,ok in zip(keys,good) if ok];raw=raw[good];y=(bugs[good]>0).astype('float32')
                idx={k:i for i,k in enumerate(keys)};edge=set();source_count=0
                for name,i in idx.items():
                    path=sources[v].get(name)
                    if path is None:continue
                    source_count+=1
                    code=z.read(path).decode('utf-8',errors='replace')
                    for imp in re.findall(r'(?m)^\s*import\s+(?!static\b)([\w.]+)\s*;',code):
                        j=idx.get(imp)
                        if j is not None and i!=j:edge.add((i,j))
                edges=np.array(sorted(edge),dtype=np.int64).reshape(-1,2)
                raw_data[v]=(keys,raw,y,edges)
                target_stats.append({'version':v,'window':window,'n':len(keys),'positive':int(y.sum()),'negative':int(len(y)-y.sum()),'edges':len(edges),'matched_java_sources':source_count,'target_labels':len(frames[v]),'strict_coverage':round(len(keys)/len(frames[v]),4),'status':'prepared'})
            if not raw_data:
                manifest['excluded'][project]={'versions':versions,'reason':'no usable three-release intersections'};continue
            train_release=next(iter(raw_data));tr=raw_data[train_release][1].reshape(-1,len(FEATURES));tr=np.where(np.isfinite(tr),tr,np.nan)
            mu=np.nanmean(tr,axis=0);mu=np.where(np.isfinite(mu),mu,0.);sd=np.nanstd(tr,axis=0);sd=np.where(np.isfinite(sd)&(sd>1e-6),sd,1.)
            projectdir=out/project;projectdir.mkdir(exist_ok=True)
            for v,(keys,raw,y,edges) in raw_data.items():
                fill=np.where(np.isfinite(raw),raw,mu)
                X=np.clip((fill-mu)/sd,-10,10).astype('float32');loc=np.nan_to_num(raw[:,-1,FEATURES.index('loc')],nan=1.,posinf=1.,neginf=1.).clip(min=1).astype('float32')
                np.savez_compressed(projectdir/f'{project}-{v}.npz',X=X,y=y,edges=edges,ids=np.array(keys,dtype=str),loc=loc)
            manifest['projects'][project]={'versions':versions,'train_release':train_release,'n_windows':len(raw_data),'targets':target_stats,'evaluation_note':('only one target window: not enough releases for temporal train/validation/test' if len(raw_data)==1 else 'chronological holdout; for only two targets, no independent validation release')}
            print(project,len(raw_data),[(x['version'],x['n'],x['positive'],x['edges']) for x in target_stats],flush=True)
        (out/'manifest_multi.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    return manifest
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--out',default='data');a=p.parse_args();main(a.archive,a.out)
