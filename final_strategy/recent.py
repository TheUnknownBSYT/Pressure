"""Less restrictive recent-form comparison and out-of-time scoring on common players."""
from final_strategy.analyse import *

def diagnostic(m,s,role,end):
    x,_=sample(m,s,role,end,eval_only=True,draws=1000)
    x['supported']=x.supported & x.hm.ge(4)&x.hb.ge(25)&x.om.ge(3)&x.ob.ge(40)
    x['eligible']=x.supported & x.absolute.gt(0)&x.uplift.gt(0)
    x['rank']=x.score.where(x.eligible).rank(ascending=False,method='min')
    return x

def run():
    d=pd.read_parquet('outputs/phase2/phase2_player_event_metrics.parquet');d,_=stratified_wickets(d);g=source_for(d,'stratified');dates=d.drop_duplicates('match_id').set_index('match_id').match_date_parsed;end=dates.max()
    bt=[]
    for role,label in ROLES.items():
        scales=json.loads((E/'settings.json').read_text())['scales'][role]
        m=match_metrics(g,role,g.high_pressure_selected);m['date']=m.match_id.map(dates)
        cut=end-pd.Timedelta(days=364)
        r=diagnostic(m[m.date>=cut],scales,role,end)
        early=diagnostic(m[m.date<cut],scales,role,cut-pd.Timedelta(days=1))
        x=pd.read_csv(E/f'{role}_all_players.csv').set_index('player_id')
        x['recent_diagnostic_rank']=r['rank'];x['recent_diagnostic_score']=r.score.where(r.supported)
        x['recent_diagnostic_hm']=r.hm;x['recent_diagnostic_hb']=r.hb;x['recent_diagnostic_positive_share']=r.positive_share.where(r.supported)
        x['earlier_diagnostic_score']=early.score.where(early.supported)
        x['combined_diagnostic_score']=((x.score+x.recent_diagnostic_score)/2).where(x.eligible.eq(True)&x.recent_diagnostic_rank.notna())
        x['combined_diagnostic_rank']=x.combined_diagnostic_score.rank(ascending=False,method='min')
        e=x.earlier_diagnostic_score;v=x.recent_diagnostic_score
        x['recent_assessment']=np.select([v.isna(),e.isna(),(e>.5)&(v>.5),(e>.5)&(v<=.5),(e<=.5)&(v>.5)],['insufficient recent comparison','recent comparison only','above midpoint in both periods','possible decline','recently improved'],default='at or below midpoint in both periods')
        x.to_csv(E/f'{role}_all_players.csv')
        sel=pd.read_csv(OUT/f'final_{label}_top10.csv').set_index('player_id');sel=x.loc[sel.index].join(sel.selection_note);sel['rank']=sel['rank'].astype(int);sel.to_csv(OUT/f'final_{label}_top10.csv')
        for year in [2024,2025,2026]:
            cut=pd.Timestamp(f'{year}-01-01');end_tr=cut-pd.Timedelta(days=1)
            threshold=d.loc[d.pressure_eligible&d.match_date_parsed.lt(cut),'pressure_score_v1'].quantile(.75)
            mm=match_metrics(g,role,d.pressure_eligible&d.pressure_score_v1.ge(threshold));mm['date']=mm.match_id.map(dates)
            tr=mm[mm.date<cut];h=tr[tr.high.eq(1)].groupby('player_id')[['rr','wr','legal']].sum().query('legal>=50');sc=h[['rr','wr']].div(h.legal,axis=0).mul(100).std().to_numpy()
            r=diagnostic(tr[tr.date>=cut-pd.Timedelta(days=365)],sc,role,end_tr)
            preds=pd.read_csv(E/f'{role}_forward_{year}.csv').set_index('player_id');preds['recent']=r.score.where(r.supported);preds['combined']=(preds.historical+preds.recent)/2;preds.dropna(inplace=True)
            for method in ['historical','recent','decay','combined','phase3_proxy']:
                bt.append(dict(role=role,year=year,method=method,n=len(preds),rho=preds[method].rank().corr(preds.observed.rank()) if len(preds)>2 else np.nan,mse=((preds[method]-preds.observed)**2).mean() if method!='phase3_proxy' else np.nan))
    pd.DataFrame(bt).to_csv(E/'recent_diagnostic_backtest.csv',index=False)
if __name__=='__main__':run()
