from final_strategy.analyse import *

def run():
    d=pd.read_parquet('outputs/phase2/phase2_player_event_metrics.parquet')
    d,_=stratified_wickets(d);g=source_for(d,'stratified');end=d.match_date_parsed.max();start=end-pd.Timedelta(days=364)
    dt=d.drop_duplicates('match_id').set_index('match_id').match_date_parsed
    scales=json.loads((E/'settings.json').read_text())['scales'];wlist=[];stage=[]
    for role,label in ROLES.items():
        pid='striker_id' if role=='bat' else 'bowler_id';m=match_metrics(g,role,g.high_pressure_selected);m['date']=m.match_id.map(dt)
        full=pd.read_csv(E/f'{role}_all_players.csv').set_index('player_id');recent=m[m.date>=start]
        # A lighter standard is useful for identifying direction, never confers a main-list place.
        light,_=sample(recent,scales[role],role,end,eval_only=True,draws=2000)
        high=recent[recent.high.eq(1)]
        full['recent_pressure_balls_all']=high.groupby('player_id').legal.sum().reindex(full.index,fill_value=0)
        full['recent_pressure_matches_all']=high[high.legal.gt(0)].groupby('player_id').match_id.nunique().reindex(full.index,fill_value=0)
        full['recent_light_score']=light.score.where(light.supported);full['recent_light_matches']=light.hm.where(light.supported)
        full['recent_light_balls']=light.hb.where(light.supported)
        full['recent_hm']=full.recent_hm.fillna(0);full['recent_hb']=full.recent_hb.fillna(0)
        full['earlier_share']=(full.early_score-.5)
        # Count direct IPL exposure including unmatched roles.
        ip=m[m.comparison_cell.str.endswith('|IPL')&m.high.eq(1)];full['IPL_pressure_all']=ip.groupby('player_id').legal.sum().reindex(full.index,fill_value=0)
        for comp in ['T20I','other_T20']:
            cc=m[m.comparison_cell.str.endswith('|'+comp)&m.high.eq(1)];full[comp+'_pressure_all']=cc.groupby('player_id').legal.sum().reindex(full.index,fill_value=0)
        annual=g[g.analysis_available&g.high_pressure_selected].groupby([pid,g.match_date_parsed.dt.year,'competition_type']).agg(legal=('legal_event','sum'),events=('delivery_id','size'),matches=('match_id','nunique')).reset_index()
        annual.rename(columns={pid:'player_id'}).to_csv(E/f'{role}_year_competition.csv',index=False)
        # Expose the ordered changes under successive, named rule changes.
        prev=pd.read_csv(f'outputs/independent_audit/{label}_corrected_diagnostic.csv').set_index('player_id')
        comp=pd.read_csv(E/f'{role}_method_comparison.csv').set_index('player_id')
        full['corrected_global_absolute_R_rank']=prev.R_rank
        full['global_repeatability_rank']=comp.global_rank
        full['phase_aware_absolute_R_rank']=full.legacy.where(full.supported.eq(True)).rank(ascending=False,method='min')
        for f in ['recent_rank','decay_rank','combined_rank','relaxed_rank','runs_only_rank']:full[f]=comp[f]
        def explain(r):
            if r.selection_reason!='selected' and pd.isna(r['rank']):return str(r.selection_reason)
            return f"With the phase-aware wicket baseline, the earlier absolute-performance formula ranks this player {r.phase_aware_absolute_R_rank:.0f}. The final rule credits {r.success_mass:.2f} successful votes from {r.effective_votes:.2f}, giving rank {r['rank']:.0f}."
        full['movement_explanation']=full.apply(explain,axis=1)
        full.to_csv(E/f'{role}_all_players.csv');sel=pd.read_csv(OUT/f'final_{label}_top10.csv').set_index('player_id')
        sel=full.loc[sel.index].join(sel.selection_note)
        sel['rank']=sel['rank'].astype(int);sel.to_csv(OUT/f'final_{label}_top10.csv')
        full[full['rank'].le(10)|full.phase3_rank.le(10)].sort_values('rank').to_csv(E/f'{role}_selection_changes.csv')
        watch=pd.read_csv(E/f'{role}_watchlist.csv');watch['role']=role;wlist.append(watch)
    pd.concat(wlist).to_csv(OUT/'under_the_radar_watchlist.csv',index=False)
if __name__=='__main__':run()
