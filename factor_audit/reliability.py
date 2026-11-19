"""Rebuild ordinary support, exposure gates and point estimates after deleting each match."""
import numpy as np
import pandas as pd

from final_strategy import ratings
from .forecast import OUT


def point(g):
    cells=ratings.supported_cells(g)
    if not cells: return np.nan,np.nan,False
    _,_,a=ratings.role_array(g,cells)
    h,w,q,mix=ratings.per_match(a[:,:,1])
    o=a[:,:,0,0]
    cell=6*a[:,:,0,1].sum(0)/o.sum(0)
    quality=np.average(q,weights=w)
    uplift=quality-np.average(mix@cell,weights=w)
    coverage=h.sum()/g.loc[g.high.eq(1),'legal'].sum()
    return quality,uplift,bool((h>0).sum()>=8 and (o.sum(1)>0).sum()>=8 and
                               h.sum()>=50 and o.sum()>=100 and coverage>=.75)


def run():
    e=pd.read_parquet(OUT/'predictions.parquet').merge(pd.read_parquet(OUT/'pressure_labels.parquet'),
        on=['delivery_id','match_id','match_date_parsed'],validate='one_to_one')
    e=e[e.competition_type.isin(['IPL','T20I'])]
    finalists=pd.read_csv(OUT/'top10_evidence.csv')
    rows,adjacent=[],[]
    for role in ['bat','bowl']:
        for version,label in [('A','original'),('G','consistent_series')]:
            grouped=ratings.matched_groups(e,role,label,role+'_published')
            if version=='G':
                end=pd.to_datetime(e.match_date_parsed.max())
                with_stability,_=ratings.evaluate_sample(grouped,end,stability=True)
                adjacent.extend(dict(role=role,model=version,**entry) for entry in with_stability.attrs['adjacent'])
            for r in finalists[(finalists.role==role)&(finalists.model==version)].itertuples():
                g=grouped[grouped.player_id.eq(r.player_id)]
                q,u,support=point(g)
                np.testing.assert_allclose([q,u],[r.absolute,r.uplift])
                assert support
                values=np.array([point(g[g.match_id.ne(mid)]) for mid in g.match_id.unique()])
                positive=(values[:,0]>0)&(values[:,1]>0)
                rows.append(dict(role=role,model=version,rank=int(r.rank),player_id=r.player_id,player_name=r.player_name,
                                 deletions=len(values),supported=int(values[:,2].sum()),positive_both=int(positive.sum()),
                                 eligible=int((positive & (values[:,2]>0)).sum()),
                                 minimum_pressure=np.nanmin(values[:,0]),minimum_uplift=np.nanmin(values[:,1])))
    pd.DataFrame(rows).to_csv(OUT/'deletion_rechecked.csv',index=False)
    pd.DataFrame(adjacent).to_csv(OUT/'recommended_adjacent.csv',index=False)


if __name__=='__main__': run()
