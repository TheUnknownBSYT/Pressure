"""Sensitivity, cross-definition validation, support profiles and provenance."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from final_strategy import ratings
from .forecast import OUT
from .rank import VARIANTS


def correlation(x,y):
    if min(len(set(x)),len(set(y)))<2: return np.nan
    return np.corrcoef(rankdata(x),rankdata(y))[0,1]


def common_targets(pairs):
    rows=[]
    own=pairs[pairs.target_type.eq('version')]
    for (year,role),z in own.groupby(['year','role']):
        for target_model in ['A','E','G']:
            target=z[z.model.eq(target_model)].set_index('player_id')
            for model in VARIANTS:
                hist=z[z.model.eq(model)].set_index('player_id')
                a=z[z.model.eq('A')].set_index('player_id')
                ids=a.index.intersection(hist.index).intersection(target.index)
                for metric in ['future_absolute','future_uplift','future_vote']:
                    x=hist.loc[ids,'rating'].to_numpy()
                    anchor=a.loc[ids,'rating'].to_numpy()
                    y=target.loc[ids,metric].to_numpy()
                    rho=correlation(x,y)
                    ref=correlation(anchor,y)
                    # Player resampling is descriptive; opponents and shared matches can couple players.
                    lo=hi=np.nan
                    if len(ids)>=6:
                        rng=np.random.default_rng(1742)
                        idx=rng.integers(0,len(ids),size=(1000,len(ids)))
                        def many(v):
                            ranks=rankdata(v[idx],axis=1)
                            ranks-=ranks.mean(axis=1,keepdims=True)
                            norm=np.linalg.norm(ranks,axis=1,keepdims=True)
                            return ranks/np.where(norm>0,norm,np.nan)
                        diffs=np.sum((many(x)-many(anchor))*many(y),axis=1)
                        lo,hi=np.nanquantile(diffs,[.025,.975])
                    rows.append(dict(role=role,year=year,model=model,target_definition=target_model,
                        target=metric,n=len(ids),rho=rho,reference_rho=ref,delta_lo=lo,delta_hi=hi))
    pd.DataFrame(rows).to_csv(OUT/'paired_player_validation.csv',index=False)


def current_baselines(e,r):
    rows=[]
    end=pd.read_csv('matches.csv',parse_dates=['match_date']).match_date.max()
    for role in ['bat','bowl']:
        g=ratings.matched_groups(e,role,'original',role+'_published')
        cohort=r[(r.model=='A')&(r.role==role)&r.supported]
        rank_rows=[]
        for row in cohort.itertuples():
            f=g[g.player_id.eq(row.player_id)]
            cells=ratings.supported_cells(f)
            _,days,arr=ratings.role_array(f,cells)
            h,weight,rate,mix=ratings.per_match(arr[:,:,1])
            prior=arr[:,:,0]
            decay=np.exp2(-(end-days).dt.days.to_numpy()/365)
            cell_rate=6*(decay@prior[:,:,1])/(decay@prior[:,:,0])
            ordinary=weight@(mix@cell_rate)/weight.sum()
            recent=(end-days).dt.days.to_numpy()<=364
            shift=row.ordinary-ordinary
            rank_rows.append(dict(role=role,player_id=row.player_id,name=row.player_name,original_rank=row.rank,
                pressure=row.absolute,ordinary=row.ordinary,recent_weighted_ordinary=ordinary,
                uplift=row.uplift,recent_baseline_uplift=row.uplift+shift,rating=row.rating,
                pressure_match_weight_last365=weight[recent].sum()/weight.sum(),
                biggest_match_weight=weight.max()/weight.sum(),
                positive_own_share=row.positive_own_share))
        by=pd.DataFrame(rank_rows).set_index('player_id')
        ok=by.pressure.gt(0) & by.recent_baseline_uplift.gt(0)
        by['new_rank']=ratings.position(by.rating,ok,by.pressure)
        rows.append(by.reset_index())
    pd.concat(rows).to_csv(OUT/'recent_ordinary.csv',index=False)


def run():
    r=pd.read_csv(OUT/'candidates.csv')
    pairs=pd.read_csv(OUT/'forward_players.csv')
    common_targets(pairs)
    e=pd.read_parquet(OUT/'predictions.parquet').merge(pd.read_parquet(OUT/'pressure_labels.parquet'),
        on=['delivery_id','match_id','match_date_parsed'],validate='one_to_one')
    e=e[e.competition_type.isin(['IPL','T20I'])]
    current_baselines(e,r)
    summaries=[]
    for (model,role),z in r.groupby(['model','role']):
        top=z[z['rank'].le(10)]
        supported=z[z.supported]
        eligible=z[z['rank'].notna()]
        original=z[z.original_rank.le(10)]
        coverage=top.pressure_legal.sum()
        summaries.append(dict(model=model,role=role,recorded_players=len(z),supported=len(supported),eligible=len(eligible),
            returned=len(top),overlap_A=int(top.original_rank.le(10).sum()),missing_originals=10-len(original),
            selected_median_matches=top.pressure_matches.median(),selected_median_balls=top.pressure_legal.median(),
            selected_no_IPL=int(top.IPL_legal.eq(0).sum()),selected_death_ball_share=top.death_legal.sum()/coverage,
            supported_death_ball_share=supported.death_legal.sum()/supported.pressure_legal.sum(),
            supported_median_matches=supported.pressure_matches.median(),supported_median_balls=supported.pressure_legal.median(),
            rank_score_volume_rho=correlation(eligible.rating,eligible.pressure_legal),
            top_median_ordinary=top.ordinary.median(),top_median_pressure=top.absolute.median(),
            top_median_uplift=top.uplift.median(),top_majority_above_own=int(top.positive_own_share.gt(.5).sum())))
    pd.DataFrame(summaries).to_csv(OUT/'model_summary.csv',index=False)
    # Independently check raw inputs and frozen release outputs against published hashes.
    book=json.loads(Path('submission/evidence/verification.json').read_text())
    hashes={}
    for filename,expected in {**{f'{k}.csv':v for k,v in book['source_hashes'].items()},
                              **{'submission/'+k:v for k,v in book['submission_sha256'].items()}}.items():
        with open(filename,'rb') as f: hashes[filename]=hashlib.file_digest(f,'sha256').hexdigest()
        assert hashes[filename]==expected, filename
    (OUT/'input_verification.json').write_text(json.dumps(hashes,indent=2)+'\n')


if __name__=='__main__': run()
