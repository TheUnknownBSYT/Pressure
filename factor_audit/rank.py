"""Score A–F on identical gates, units, resampling and temporal splits. Keep A as the common future target."""
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from final_strategy import ratings
from phase2_pipeline.core import identities
from .forecast import OUT

# E isolates the stage term; F tests adding lagged personal form to that same specification.
VARIANTS = {
    'A': ('published','original',None), 'B': ('ipl_only','original',None),
    'C': ('competition','original',None), 'D': ('both','original',None),
    'E': ('published','no_stakes',None), 'F': ('form','no_stakes',None),
    'pooled_IPL': ('published','original',None), 'basic': ('no_opp','original',None),
    'stage_interaction': ('published','interaction',None), 'half_stage': ('published','half_bonus',None),
    'leverage': ('published','leverage_only',None), 'form_only': ('form','original',None),
    'decay365': ('published','original',365), 'home': ('home','original',None),
    'opponent_decay': ('decayed','original',None), 'G': ('published','consistent_series',None),
    'series_mid': ('published','series_mid',None), 'series_league': ('published','series_league',None),
}


def evaluate(g,end,half):
    mids=np.sort(g.match_id.unique())
    counts=ratings.counts_matrix(mids,2000)
    rows,ordinary=[],{}
    for pid,frame in g.groupby('player_id'):
        # Ratings are available from 4 matches/20 balls. Keep all players for coverage diagnostics.
        row,ordinary[pid],_=ratings.estimate_player(frame,end,mids,counts,half_life=half)
        rows.append(dict(player_id=pid,**row))
    result=pd.DataFrame(rows).set_index('player_id')
    for col in ['rating','absolute','uplift','pressure_coverage']:
        if col not in result: result[col]=np.nan
    ok=result.supported & result.absolute.gt(0) & result.uplift.gt(0)
    result['rank']=ratings.position(result.rating,ok,result.absolute)
    return result,ordinary


def gates(r):
    for label,matches,balls,ordinary,coverage in [('relaxed',6,40,80,.65),('published',8,50,100,.75),
                                                  ('strict',10,75,125,.85)]:
        mask=(r.pressure_matches.ge(matches) & r.ordinary_matches.ge(matches) & r.pressure_legal.ge(balls) &
              r.ordinary_legal.ge(ordinary) & r.pressure_coverage.ge(coverage) &
              r.absolute.gt(0) & r.uplift.gt(0) & r.rating.notna())
        yield label,ratings.position(r.rating,mask,r.absolute)


def run():
    e=pd.read_parquet(OUT/'predictions.parquet').merge(pd.read_parquet(OUT/'pressure_labels.parquet'),
        on=['delivery_id','match_id','match_date_parsed'],validate='one_to_one')
    e=e[e.competition_type.isin(['IPL','T20I'])]
    names=identities(ratings.Path('.'))[0].set_index('player_id').player_name
    end=pd.read_csv('matches.csv',parse_dates=['match_date']).match_date.max()
    candidates,gate_rows,predictions=[],[],[]
    for role in ['bat','bowl']:
        common_targets={}
        for name,(forecast,high,half) in VARIANTS.items():
            subset=e[e.competition_type.eq('IPL')] if name in ['B','pooled_IPL'] else e
            g=ratings.matched_groups(subset,role,high,role+'_'+forecast)
            result,_=evaluate(g,end,half)
            result['player_name']=names
            result['selection_reason']=result.selection_reason.mask(result['rank'].le(10),'selected')
            candidates.append(result.reset_index().assign(role=role,model=name))
            for label,ranks in gates(result):
                if label=='published':
                    pd.testing.assert_series_equal(result['rank'].dropna().sort_index(),ranks.astype(float).sort_index(),check_names=False)
                gate_rows.append(dict(model=name,role=role,gate=label,eligible=len(ranks),returned=min(10,len(ranks)),
                                 retained=len(set(ranks[ranks<=10].index)&set(result[result['rank']<=10].index)),
                                 names=' | '.join(names.loc[ranks[ranks<=10].index])))
            if name=='A':
                archive=pd.read_csv('submission/final_'+ratings.ROLE_LABEL[role]+'_top10.csv')
                assert list(result.sort_values('rank').head(10).index)==archive.player_id.tolist()
                ref=pd.read_csv('submission/evidence/candidate_metrics.csv').query('role==@role').set_index('player_id')
                r=result.loc[ref.index]
                for col in ['absolute','uplift','rating','rank','pressure_legal','ordinary_legal']:
                    np.testing.assert_allclose(r[col],ref[col],equal_nan=True,atol=1e-9)
            for year in [2025,2026]:
                cut=pd.Timestamp(year,1,1)
                history,ord_rates=evaluate(g[g.date<cut],cut-pd.Timedelta(days=1),half)
                future=g[g.date.dt.year.eq(year)]
                own=ratings.future_targets(future,ord_rates)
                if name=='A': common_targets[year]=own
                for target_type,observed in [('version',own),('common_A',common_targets[year])]:
                    if observed.empty: continue
                    pairs=history.loc[history.supported,['rating','absolute','uplift']].join(observed,how='inner')
                    predictions.append(pairs.reset_index().assign(role=role,model=name,year=year,target_type=target_type))
            print(role,name,flush=True)
    r=pd.concat(candidates,ignore_index=True)
    key=['role','player_id']
    anchor=r[r.model.eq('A')][key+['rank','pressure_legal','pressure_matches','absolute','ordinary','uplift','rating']].set_index(key)
    r=r.join(anchor.add_prefix('original_'),on=key)
    r['rank_gain']=r.original_rank-r['rank']
    r.to_csv(OUT/'candidates.csv',index=False)
    top=r[r['rank'].le(10)].sort_values(['model','role','rank'])
    top.to_csv(OUT/'top10_evidence.csv',index=False)
    top.pivot(index=['role','rank'],columns='model',values='player_name').to_csv(OUT/'rankings_side_by_side.csv')
    pd.DataFrame(gate_rows).to_csv(OUT/'gates.csv',index=False)
    pairs=pd.concat(predictions,ignore_index=True)
    pairs.to_csv(OUT/'forward_players.csv',index=False)
    scores=[]
    for (role,year,target_type),block in pairs.groupby(['role','year','target_type']):
        common=set.intersection(*(set(block.loc[block.model.eq(m),'player_id']) for m in 'ABCDEF'))
        for name,z in block.groupby('model'):
            for cohort,s in [('available',z),('shared_A_to_F',z[z.player_id.isin(common)])]:
                for target in ['future_absolute','future_uplift','future_vote']:
                    rho=s.rating.corr(s[target],method='spearman') if len(s)>2 else np.nan
                    scores.append(dict(role=role,model=name,year=year,target_type=target_type,cohort=cohort,
                        target=target,players=len(s),rho=rho,mae=(s.rating-s[target]).abs().mean() if target=='future_absolute' else np.nan))
    pd.DataFrame(scores).to_csv(OUT/'forward_validation.csv',index=False)


if __name__=='__main__':
    with threadpool_limits(limits=1): run()
