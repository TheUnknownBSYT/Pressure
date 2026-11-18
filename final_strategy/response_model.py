"""Pre-delivery forecasts, opportunity and memory experiments. Run from project root.

python -m final_strategy.response_model
"""
from __future__ import annotations
from collections import defaultdict, deque
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import LogisticRegression, Ridge
from threadpoolctl import threadpool_limits
from phase2_pipeline.core import attribute

ROOT = Path.cwd()
OUT = ROOT / 'outputs/final_submission/evidence'
DATE = 'match_date_parsed'
FOLDS = [('2022-10-01','2023-01-01'), ('2023-01-01','2024-01-01'),
         ('2024-01-01','2025-01-01'), ('2025-01-01','2026-01-01'), ('2026-01-01','2027-01-01')]
IDS = {'bat':'striker_id','bowl':'bowler_id'}
TEAMS = {'bat':'batting_team_id','bowl':'bowling_team_id'}
OTHER = {'bat':'bowl','bowl':'bat'}
WICKET = {'bat':'bat_dismissal','bowl':'bowl_wicket'}
BASE = ['chase','ipl','powerplay','death','balls_left','resources','demand','current_rate']
ROLE = ['position','invalid_position','set_batter','early_bowler','left_hand','spin','left_spin','unknown_hand','unknown_style',
        'death_resources','chase_demand_death','venue_run','role_phase_share']
OPP = ['opponent_run','opponent_wicket','opponent_team_run','opponent_team_wicket']
ABILITY = ['own_run','own_wicket','pressure','pressure_missing']
GROUPS = {'boundaries':['after_two_boundaries','after_two_sixes'], 'dots':['after_two_dots'],
          'collapse':['recent_two_wickets','collapse'], 'momentum':['acceleration','bat_accel','spell_economy','rrr_change'],
          'form':['form_trend','strong_three'], 'slump':['poor_three'],
          'opportunity':['xi_gap','xi_appearances_90d','xi_share','xi_unknown','xi_new'],
          'recovery':['recent_failure','past_rebound','failure_rebound']}
INTER = ['pressure_x_trend','pressure_x_dots','pressure_x_boundaries','pressure_x_collapse','pressure_x_slump']
COLS = ('match_id innings innings_ball_num delivery_id match_date_parsed striker_id non_striker_id bowler_id '
        'striker_name bowler_name batting_team_id bowling_team_id competition_type standard_t20_eligible '
        'pressure_eligible pressure_score_v1 is_wicket dismissed_player_id extras_type runs '
        'bowler_runs_event_assumed bowler_wicket_event bowler_concession_uncertain phase_pre wickets_remaining_pre '
        'required_run_rate_pre legal_event legal_balls_remaining_pre current_run_rate_pre batting_order '
        'batter_legal_balls_pre bowler_legal_balls_pre batting_type pace_or_spin venue_id '
        'is_four is_six team_runs_event recent_wickets_12_pre collapse_pre team_acceleration_pre '
        'batter_acceleration_pre bowler_economy_assumed_pre required_run_rate_change_pre score_pre '
        'overid over_ball_num').split()


def date_prior(d, keys, targets):
    """Sum by date before taking lag. Every member of a same-day doubleheader sees the same history."""
    daily = d.groupby(keys+[DATE], as_index=False, sort=True)[targets].sum()
    cumulative = daily.groupby(keys, sort=False)[targets].cumsum() - daily[targets]
    return daily[keys+[DATE]].join(cumulative.add_prefix('hist_'))


def attach(d, t, keys):
    return d[keys].merge(t, on=keys, how='left', sort=False, validate='many_to_one').set_axis(d.index)


def xi_history(d, role):
    xi = pd.read_csv(ROOT/'playing_eleven.csv')
    matches = pd.read_csv(ROOT/'matches.csv',parse_dates=['match_date'])
    played = xi[xi.is_played].groupby(['match_id','team_id']).player_id.agg(set).to_dict()
    dates = defaultdict(list)
    for row in matches.itertuples():
        for team in [row.home_team_id,row.away_team_id]:
            dates[row.match_date].append((row.match_id,team))
    idcol, teamcol = IDS[role], TEAMS[role]
    requests = d[[DATE,'match_id',teamcol,idcol]].drop_duplicates().groupby(DATE)
    recent = defaultdict(lambda:deque(maxlen=5))
    appearances = defaultdict(list)
    rows = []
    for day in sorted(dates):
        if day in requests.groups:
            for _, mid, team, pid in requests.get_group(day).itertuples(index=False,name=None):
                hist = recent[team]; prev = appearances[pid]
                usable = len(hist)==5 and all(record is not None for record in hist)
                rows.append((mid,team,pid,
                             np.log1p((day-prev[-1]).days) if prev else 0,
                             sum((day-date).days <= 90 for date in prev),
                             sum(pid in record for record in hist)/5 if usable else 0,
                             int(not usable), int(not prev)))
        today = set()
        for mid, team in sorted(dates[day]):
            roster = played.get((mid,team),set())
            good = 11 <= len(roster) <= 12
            recent[team].append(roster if good else None)
            if good: today.update(roster)
        for pid in today: appearances[pid].append(day)
    table = pd.DataFrame(rows,columns=['match_id',teamcol,idcol]+GROUPS['opportunity'])
    return attach(d,table,['match_id',teamcol,idcol])[GROUPS['opportunity']]


def long_history(d, role):
    """Run-form residuals use prior-date competition/phase tempo and the player's own earlier career rate."""
    pid,team = IDS[role],TEAMS[role]
    a = d[[DATE,'match_id',pid,team,'venue_id','competition_type','phase_pre','innings']].copy()
    a['n'] = 1
    a['r'] = d[role+'_runs'].astype(float)
    a['w'] = d[WICKET[role]].astype(float)
    cells = ['competition_type','phase_pre','innings']
    h = attach(a,date_prior(a,cells,['n','r','w']),cells+[DATE])
    # Fixed startup priors contribute 60 events and vanish with support.
    a['r'] -= (h.hist_r+60*1.25)/(h.hist_n+60)
    a['w'] -= (h.hist_w+60*.05)/(h.hist_n+60)
    hist = pd.DataFrame(index=d.index)
    for key,label,prior in [(pid,'individual',100),(team,'team',500),('venue_id','venue',500)]:
        h = attach(a,date_prior(a,[key],['n','r','w']),[key,DATE])
        hist[label+'_run'] = h.hist_r/(h.hist_n+prior)
        hist[label+'_wicket'] = h.hist_w/(h.hist_n+prior)
        if key==pid: total = h.hist_n
    phase = attach(a,date_prior(a,[pid,'phase_pre'],['n']),[pid,'phase_pre',DATE])
    hist['role_phase_share'] = (phase.hist_n+10)/(total+30)
    daily = a.groupby([pid,DATE],as_index=False).agg(r=('r','sum'),n=('n','sum'),matches=('match_id','nunique'))
    g = daily.groupby(pid,sort=False)
    signed = 1 if role=='bat' else -1
    prior_matches = g.matches.cumsum()-daily.matches
    prior = (g.r.cumsum()-daily.r)/(g.n.cumsum()-daily.n+100)
    daily['performance'] = signed*(daily.r - daily.n*prior)/(daily.n+6)
    # One observation per player-date. Same-date performances form a single history entry.
    daily['poor'] = (daily.performance<0) & (prior_matches>=3)
    daily['strong'] = (daily.performance>0) & (prior_matches>=3)
    daily['recent_failure'] = g.poor.shift(1).fillna(False).astype(float)
    daily['poor_three'] = daily.recent_failure.astype(bool) & g.poor.shift(2).fillna(False) & g.poor.shift(3).fillna(False)
    daily['strong_three'] = g.strong.shift().fillna(False) & g.strong.shift(2).fillna(False) & g.strong.shift(3).fillna(False)
    daily['form_trend'] = g.performance.transform(lambda s:s.shift().rolling(5,min_periods=2).mean()).fillna(0)
    daily['rebound_value'] = daily.recent_failure*daily.performance
    daily['past_rebound'] = (g.rebound_value.cumsum()-daily.rebound_value)/(g.recent_failure.cumsum()-daily.recent_failure+8)
    daily['failure_rebound'] = daily.recent_failure*daily.past_rebound
    feats = GROUPS['form']+GROUPS['slump']+GROUPS['recovery']
    h = attach(d,daily[[pid,DATE]+feats],[pid,DATE])
    return pd.concat([hist,h[feats],xi_history(d,role)],axis=1), daily


def entry_position(d):
    # Count identities seen at the crease up to this event. Simultaneous debuts
    # (typically the two openers) receive the same average entry position.
    ids = ['match_id','innings','innings_ball_num']
    arrivals = pd.concat([d[ids+[key]].rename(columns={key:'pid'}) for key in ['striker_id','non_striker_id']])
    arrivals = arrivals.groupby(['match_id','innings','pid'],as_index=False).innings_ball_num.min()
    arrivals['position'] = arrivals.groupby(['match_id','innings']).innings_ball_num.rank(method='average')
    z = arrivals.rename(columns={'pid':'striker_id'})
    return attach(d,z[['match_id','innings','striker_id','position']],['match_id','innings','striker_id']).position


def prior_style(d, pid, col):
    # Only labels recorded on a strictly earlier date may establish the style.
    observed = d[[pid,DATE,col]].dropna().sort_values(DATE).drop_duplicates([pid,DATE],keep='last')
    observed[col] = observed.groupby(pid)[col].shift()
    return attach(d,observed,[pid,DATE])[col].fillna('unknown')


def build():
    raw = pd.read_parquet(ROOT/'outputs/pressure_scored.parquet',columns=COLS)
    assert raw.delivery_id.is_unique
    d = attribute(raw[raw.standard_t20_eligible].sort_values([DATE,'match_id','innings','innings_ball_num']).reset_index(drop=True))
    d['innings_total'] = d.groupby(['match_id','innings']).team_runs_event.transform('sum')
    dot = d.legal_event.eq(1) & d.team_runs_event.eq(0) & ~d.is_wicket
    boundary = d.is_four | d.is_six
    for col, cond, key in [('after_two_dots',dot,'striker_id'),('after_two_boundaries',boundary,'bowler_id'),('after_two_sixes',d.is_six,'bowler_id')]:
        g = cond.groupby([d.match_id,d.innings,d[key]],sort=False)
        d[col] = g.shift(1).fillna(False) & g.shift(2).fillna(False)
    d['observed_position'] = entry_position(d)
    d['historical_hand'] = prior_style(d,'striker_id','batting_type')
    d['historical_style'] = prior_style(d,'bowler_id','pace_or_spin')
    d = d[d.outcome_valid].reset_index(drop=True)
    h, daily = {},{}
    for r in IDS: h[r], daily[r] = long_history(d,r)
    x = pd.DataFrame(index=d.index)
    x['chase'] = d.innings.eq(2); x['ipl'] = d.competition_type.eq('IPL')
    x['powerplay'] = d.phase_pre.eq('powerplay'); x['death'] = d.phase_pre.eq('death')
    x['balls_left'] = d.legal_balls_remaining_pre/120; x['resources'] = d.wickets_remaining_pre/10
    x['demand'] = d.required_run_rate_pre.clip(0,36).fillna(0)/12
    x['current_rate'] = d.current_run_rate_pre.clip(0,24).fillna(0)/12
    x['invalid_position'] = ~d.observed_position.between(1,11); x['position'] = d.observed_position.where(~x.invalid_position,6)/11
    x['set_batter'] = np.log1p(d.batter_legal_balls_pre); x['early_bowler'] = d.bowler_legal_balls_pre.lt(6)
    x['left_hand'] = d.historical_hand.eq('LHB'); x['spin'] = d.historical_style.str.lower().eq('spin')
    x['unknown_hand'] = d.historical_hand.eq('unknown'); x['unknown_style'] = d.historical_style.eq('unknown')
    x['left_spin'] = x.left_hand & x.spin; x['death_resources'] = x.death*x.resources
    x['chase_demand_death'] = x.death*x.demand
    x['pressure'] = d.pressure_score_v1.fillna(0)/100; x['pressure_missing'] = d.pressure_score_v1.isna()
    for name in ['after_two_boundaries','after_two_sixes','after_two_dots']: x[name]=d[name]
    x['recent_two_wickets'] = d.recent_wickets_12_pre.ge(2); x['collapse'] = d.collapse_pre
    x['acceleration'] = d.team_acceleration_pre.clip(-18,18).fillna(0)
    x['bat_accel'] = d.batter_acceleration_pre.clip(-300,300).fillna(0)/100
    x['spell_economy'] = d.bowler_economy_assumed_pre.clip(0,36).fillna(0)/12
    x['rrr_change'] = d.required_run_rate_change_pre.clip(-12,12).fillna(0)
    features = {}
    for r in IDS:
        xr = x.copy()
        o = OTHER[r]
        xr['role_phase_share'] = h[r].role_phase_share
        xr['venue_run'] = h['bat'].venue_run
        for outcome in ['run','wicket']:
            xr['opponent_'+outcome] = h[o]['individual_'+outcome]
            xr['opponent_team_'+outcome] = h[o]['team_'+outcome]
            xr['own_'+outcome] = h[r]['individual_'+outcome]
        for k in ['form','slump','recovery','opportunity']:
            for c in GROUPS[k]: xr[c] = h[r][c]
        for col,name in zip(INTER,['form_trend','after_two_dots','after_two_boundaries','collapse','poor_three']):
            xr[col] = xr.pressure*xr[name]
        assert np.isfinite(xr.astype(float).to_numpy()).all()
        features[r] = xr.astype(float)
    return d,features,daily


def resource_basis(b,w):
    b = np.asarray(b)/120
    kernels = np.stack([b, np.sqrt(b), b*b],axis=1)
    return ((np.arange(1,11)[None,:,None] <= np.asarray(w)[:,None,None])*kernels[:,None,:]).reshape(-1,30)


def wicket_prices(d,tr,te):
    # Completed first innings as training targets. Aggregate identical states with
    # equal total weight per innings, then fit the nonnegative marginal surfaces.
    m = d.loc[tr & d.innings.eq(1)].copy()
    m['weight'] = 1/m.groupby('match_id').delivery_id.transform('size')
    m['weighted_future'] = (m.innings_total-m.score_pre)*m.weight
    s = m.groupby(['legal_balls_remaining_pre','wickets_remaining_pre'],as_index=False).agg(
        mass=('weight','sum'),y=('weighted_future','sum'))
    basis = resource_basis(s.legal_balls_remaining_pre,s.wickets_remaining_pre)
    # Smooth adjacent wickets and penalize weakly determined coefficients.
    eye = np.eye(30)
    X = np.vstack([basis*np.sqrt(s.mass.to_numpy())[:,None], .2*(eye[:-3]-eye[3:]), .05*eye])
    y = np.r_[s.y/np.sqrt(s.mass),np.zeros(57)]
    coef = nnls(X,y,maxiter=2000)[0]
    t=d.loc[te]; balls=t.legal_balls_remaining_pre.to_numpy(); wickets=t.wickets_remaining_pre.to_numpy()
    # Price before observing delivery legality. Use the one-legal-ball transition for every event.
    full = resource_basis(balls,wickets)@coef
    b=np.maximum(balls-1,0)
    price = (resource_basis(b,wickets)-resource_basis(b,np.maximum(wickets-1,0)))@coef
    first=t.innings.eq(1)
    future = (t.innings_total-t.score_pre).to_numpy()
    naive_rate = m.team_runs_event.sum()/m.legal_event.sum()
    return price,coef,np.sqrt(np.mean((future[first]-full[first])**2)),np.sqrt(np.mean((future[first]-balls[first]*naive_rate)**2))


def choose_models():
    models={'mean':[], 'context':BASE, 'role':BASE+ROLE,'opposition':BASE+ROLE+OPP}
    anchor = models['opposition']+ABILITY
    models['ability']=anchor
    for g,cols in GROUPS.items(): models[g]=anchor+cols
    models['all_memory'] = anchor + sum(GROUPS.values(),[])
    models['interactions'] = models['all_memory']+INTER
    return models


def legal_probability(d, tr, t):
    # Historical conversion between recorded-event expectations and legal exposure.
    # Partial pooling keeps unsupported competition/phase/innings cells finite.
    keys = ['competition_type','phase_pre','innings']
    history = d.loc[tr]
    cell = history.groupby(keys).legal_event.agg(['sum','size'])
    cell['probability'] = (cell['sum']+100*history.legal_event.mean())/(cell['size']+100)
    return t[keys].join(cell.probability,on=keys).probability.fillna(history.legal_event.mean()).to_numpy()


def model_fit(X, run, wicket, tr, te, cols):
    if not cols:
        return np.full(te.sum(),run[tr].mean()), np.full(te.sum(),wicket[tr].mean()), []
    names=X.columns.get_indexer(cols)
    a = X.to_numpy()[tr][:,names].T.copy().T
    b = X.to_numpy()[te][:,names].T.copy().T
    mean=a.mean(axis=0); scale=a.std(axis=0); scale[scale<1e-6]=1
    a=(a-mean)/scale; b=(b-mean)/scale
    runmod = Ridge(alpha=200,solver='cholesky').fit(a,run[tr])
    wmod = LogisticRegression(solver='newton-cholesky',C=.02,max_iter=40,tol=1e-6).fit(a,wicket[tr])
    rr=np.maximum(0,runmod.predict(b)); pp=wmod.predict_proba(b)[:,1]
    params=[dict(feature=c,train_mean=m,train_sd=s,run_coefficient=r,wicket_logodds_coefficient=w) for c,m,s,r,w in zip(cols,mean,scale,runmod.coef_,wmod.coef_[0])]
    params.append(dict(feature='intercept',train_mean=0,train_sd=1,run_coefficient=runmod.intercept_,wicket_logodds_coefficient=wmod.intercept_[0]))
    return rr,pp,params


def errors(t, rr, pp, r):
    y = t[r+'_runs'].to_numpy()
    w = t[WICKET[r]].to_numpy()
    return pd.DataFrame({'match_id':t.match_id.to_numpy(),'squared_run_error':(rr-y)**2,
        'brier':(pp-w)**2,'wicket_log_loss':-np.log(np.where(w,pp,1-pp).clip(1e-9)),
        'expected_runs':rr,'observed_runs':y,'expected_wickets':pp,'observed_wickets':w})


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    d,x,daily=build()
    specs=choose_models()
    logs=[]; metrics=[]; params=[]; calibration=[]; prices=[]; segments=[]; loss=[]; examples=[]
    chunks = []
    def log(msg):
        print(msg,flush=True)
        logs.append(msg)
    log('Ridge runs alpha=200; logistic wicket C=.02; coefficients per training SD. Working unit is recorded event.')
    for group in [True, False]:
        for start,end in FOLDS:
            tr=d[DATE].lt(start).to_numpy()
            te=(d[DATE].ge(start)&d[DATE].lt(end)&d.pressure_eligible).to_numpy()
            assert d.loc[tr,DATE].max()<d.loc[te,DATE].min()
            assert set(d.loc[tr,'match_id']).isdisjoint(d.loc[te,'match_id'])
            t=d[te].reset_index(drop=True)
            history_pressure=d.loc[tr & d.pressure_eligible,'pressure_score_v1']
            high_threshold=history_pressure.quantile(.75)
            high=t.pressure_score_v1.ge(high_threshold).to_numpy()
            if group:
                price,c,rmse,naive_rmse=wicket_prices(d,tr,te)
                prices.extend(dict(test_start=start,remaining_wicket=j+1,kernel=k,coefficient=v,heldout_future_runs_rmse=rmse,constant_run_rate_rmse=naive_rmse)
                    for j,vals in enumerate(c.reshape(10,3)) for k,v in zip(['linear','sqrt','quadratic'],vals))
                e=t[['delivery_id','match_id',DATE,'legal_event']].copy()
                e['wicket_price']=price
                e['historical_legal_probability']=legal_probability(d,tr,t)
                exposure=t.legal_event.to_numpy()/e.historical_legal_probability.to_numpy()
            for r in IDS:
                for model,cols in specs.items():
                    if (model in ['role','opposition']) != group: continue
                    run_y=d[r+'_runs'].to_numpy(); w_y=d[WICKET[r]].to_numpy()
                    rr,pp,pars=model_fit(x[r],run_y,w_y,tr,te,cols)
                    params.extend(dict(test_start=start,role=r,model=model,**p) for p in pars)
                    lf=errors(t,rr,pp,r)
                    m=lf.drop(columns='match_id').mean().to_dict()
                    metrics.append(dict(test_start=start,sample='all',role=r,model=model,
                        threshold=high_threshold,events=len(t),matches=t.match_id.nunique(),**m))
                    if high.any():
                        metrics.append(dict(test_start=start,sample='high',role=r,model=model,threshold=high_threshold,
                            events=high.sum(),matches=t.loc[high,'match_id'].nunique(),**lf.loc[high].drop(columns='match_id').mean().to_dict()))
                    if model in ['context','role','opposition','ability','all_memory','interactions']:
                        for by in ['competition_type','phase_pre','innings']:
                            for val,idx in t.groupby(by).indices.items():
                                segments.append(dict(test_start=start,role=r,model=model,segment=by,value=val,events=len(idx),
                                    **lf.iloc[idx].drop(columns='match_id').mean().to_dict()))
                        bins=pd.qcut(pp,10,duplicates='drop')
                        for i,(_,a) in enumerate(lf.groupby(bins,observed=True)):
                            calibration.append(dict(test_start=start,role=r,model=model,decile=i+1,events=len(a),
                                predicted=a.expected_wickets.mean(),observed=a.observed_wickets.mean()))
                    ml=lf.groupby('match_id')[['squared_run_error','brier']].mean().reset_index()
                    ml['test_start']=start; ml['role']=r;ml['model']=model;loss.append(ml)
                    if high.any():
                        ml=lf.loc[high].groupby('match_id')[['squared_run_error','brier']].mean().reset_index()
                        ml['test_start']=start;ml['role']=r;ml['model']=model;ml['high']=True;loss.append(ml)
                    log(f'{r} {model}: train {d.loc[tr,DATE].min().date()}..{d.loc[tr,DATE].max().date()} '
                        f'{d.loc[tr,"match_id"].nunique()} matches/{sum(tr)} events; test {t[DATE].min().date()}..{t[DATE].max().date()} '
                        f'{t.match_id.nunique()} matches/{len(t)} events; run MSE {m["squared_run_error"]:.6f}, wicket Brier {m["brier"]:.6f}. '
                        f'Features: {", ".join(cols) or "intercept only"}.')
                    if group:
                        sign=1 if r=='bat' else -1
                        value=sign*(t[r+'_runs'].to_numpy()-rr*exposure+price*(pp*exposure-t[WICKET[r]].to_numpy()))
                        if model=='opposition':
                            e[r+'_value']=value;e[r+'_expected_runs']=rr;e[r+'_expected_wicket']=pp
                        else: e[r+'_no_opponent_value']=value
                    if not group and model in ['ability','interactions'] and start.startswith('2026'):
                        f=t[['delivery_id','match_id',DATE,IDS[r], 'striker_name' if r=='bat' else 'bowler_name','innings','overid','over_ball_num',
                            'innings_ball_num','pressure_score_v1',r+'_runs',WICKET[r]]].copy()
                        f=f.rename(columns={IDS[r]:'player_id','striker_name' if r=='bat' else 'bowler_name':'player_name'})
                        for feature in ['poor_three','strong_three','form_trend','recent_failure','past_rebound','after_two_sixes',
                                        'after_two_boundaries','after_two_dots','collapse','xi_share','xi_unknown']:
                            f[feature]=x[r].loc[te,feature].to_numpy()
                        f['high']=high;f['role']=r;f['model']=model;f['expected_runs']=rr;f['wicket_prob']=pp;examples.append(f)
            if group: chunks.append(e)
        if group:
            event=pd.concat(chunks,ignore_index=True);assert event.delivery_id.is_unique
            event.to_parquet(OUT/'response_events.parquet',index=False)
            print('Wrote response_events.parquet',flush=True)
    def save(frame,name):
        frame.to_csv(OUT/('response_'+name+'.csv'),index=False)
    save(pd.DataFrame(metrics),'metrics');save(pd.DataFrame(params),'coefficients');save(pd.DataFrame(prices),'wicket_prices')
    save(pd.DataFrame(segments),'segments');save(pd.DataFrame(calibration),'calibration')
    losses=pd.concat(loss,ignore_index=True); losses['high']=losses.high.fillna(False)
    summary=paired_results(losses)
    save(summary,'ablation')
    for row in summary[(summary['sample']=='all')&(summary.period=='2026 reserved')].itertuples():
        log(f'{row.role} {row.model} against {row.reference}: held-out match-mean MSE delta {row.run_delta:.6f} '
            f'[{row.run_lo:.6f},{row.run_hi:.6f}], Brier delta {row.brier_delta:.7f} [{row.brier_lo:.7f},{row.brier_hi:.7f}]')
    (OUT/'response_training.log').write_text('\n'.join(logs)+'\n')
    ex=pd.concat(examples,ignore_index=True)
    write_examples(ex,daily,d,save,pd.DataFrame(params))
    profiles(d,event,save)
    coverage=[]
    for role in IDS:
        for year in [2023,2024,2025,2026]:
            ix=d.pressure_eligible & d[DATE].dt.year.eq(year)
            feats=x[role].loc[ix]
            coverage.append(dict(role=role,year=year,events=len(feats),five_prior_rosters=int(feats.xi_unknown.eq(0).sum()),
                previous_XI_known=int(feats.xi_new.eq(0).sum()),three_poor=int(feats.poor_three.sum()),
                three_strong=int(feats.strong_three.sum()),recent_selection_mean=feats.loc[feats.xi_unknown.eq(0),'xi_share'].mean()))
    save(pd.DataFrame(coverage),'history_coverage')
    write_report(pd.DataFrame(metrics),summary,event,pd.DataFrame(prices),pd.DataFrame(coverage))
    print('Finished response experiments.',flush=True)


def paired_results(losses):
    rng=np.random.default_rng(261118)
    rows=[]
    for (role,high),g in losses.groupby(['role','high']):
        for period,part in [('2023–25 development',g[g.test_start.between('2023','2025-12-31')]),('2026 reserved',g[g.test_start.eq('2026-01-01')])]:
            matrices={v:part.pivot(index='match_id',columns='model',values=v) for v in ['squared_run_error','brier']}
            for model in choose_models():
                if model=='mean':continue
                ref={'context':'mean','role':'context','opposition':'role','ability':'opposition','interactions':'all_memory'}.get(model,'ability')
                diff=np.column_stack([v[model]-v[ref] for v in matrices.values()])
                draws=diff[rng.integers(0,len(diff),(1500,len(diff)))].mean(axis=1)
                mean=diff.mean(axis=0);lo,hi=np.quantile(draws,[.025,.975],axis=0)
                rows.append(dict(role=role,sample='high' if high else 'all',period=period,model=model,reference=ref,
                    matches=len(diff),run_delta=mean[0],run_lo=lo[0],run_hi=hi[0],brier_delta=mean[1],brier_lo=lo[1],brier_hi=hi[1]))
    return pd.DataFrame(rows)


def write_examples(ex,daily,d,save,params):
    base=ex[ex.model.eq('ability')][['role','delivery_id','expected_runs','wicket_prob']].rename(columns={'expected_runs':'baseline_runs','wicket_prob':'baseline_wicket'})
    pred=ex[ex.model.eq('interactions')].merge(base,on=['delivery_id','role'],validate='one_to_one')
    # Same fitted model with just the indicated flag(s) switched off. This is
    # a coefficient decomposition, not a plausible intervention on a match.
    for r,flag,subgroup in [('bat','slump',['poor_three','pressure_x_slump']),
                            ('bowl','boundary',['after_two_sixes','after_two_boundaries','pressure_x_boundaries'])]:
        f=pred[pred.role.eq(r)]
        pc=params[(params.role==r)&params.test_start.eq('2026-01-01')&params.model.eq('interactions')].set_index('feature')
        deltar, deltaw = np.zeros(len(f)), np.zeros(len(f))
        for col in subgroup:
            value=f[col] if col in f else f.pressure_score_v1/100*f['poor_three' if r=='bat' else 'after_two_boundaries']
            deltar+=value.to_numpy()*pc.loc[col,'run_coefficient']/pc.loc[col,'train_sd']
            deltaw+=value.to_numpy()*pc.loc[col,'wicket_logodds_coefficient']/pc.loc[col,'train_sd']
        pred.loc[f.index,'flag_off_expected_runs']=np.maximum(0,f.expected_runs-deltar)
        odds=(f.wicket_prob/(1-f.wicket_prob))*np.exp(-deltaw)
        pred.loc[f.index,'flag_off_wicket_prob']=odds/(1+odds)
    # Deterministically select first reserved high-pressure ball of each type.
    short=pred[(pred.role=='bowl')&(pred.after_two_sixes>0)&pred.high].sort_values([DATE,'match_id','innings','innings_ball_num']).head(1)
    across=pred[(pred.role=='bat')&(pred.poor_three>0)&pred.high].sort_values([DATE,'match_id','innings','innings_ball_num']).head(1)
    for frame,name,k in [(short,'short','bowl'),(across,'across','bat')]:
        if frame.empty: continue
        v=frame.iloc[0]
        if name=='short':
            pre=d[(d.match_id==v.match_id)&(d.innings==v.innings)&(d.bowler_id==v.player_id)&(d.innings_ball_num<=v.innings_ball_num)].tail(3)
            ids=pre.delivery_id.tolist()
            sub=pred[(pred.role==k)&pred.delivery_id.isin(ids)].merge(pre[['delivery_id','is_four','is_six','team_runs_event','score_pre','wickets_remaining_pre']],on='delivery_id')
            save(sub,'example_short')
        else:
            save(frame,'example_across')
            hist=daily[k][(daily[k][IDS[k]]==v.player_id)&(daily[k][DATE]<=v[DATE])].tail(6).copy()
            mids=d[d[IDS[k]]==v.player_id].groupby(DATE).match_id.agg(lambda s:'|'.join(map(str,s.unique())))
            hist['match_ids']=hist[DATE].map(mids)
            save(hist,'example_across_history')


def profiles(d,event,save):
    # Descriptive clustered exposure and shrinkage for players after specific setbacks.
    v=d.merge(event[['delivery_id','bat_value','bowl_value','bat_expected_runs','bowl_expected_runs','bat_expected_wicket','bowl_expected_wicket']],on='delivery_id',validate='one_to_one')
    rows=[]
    for r,trigger in [('bat','after_two_dots'),('bat','collapse_pre'),('bowl','after_two_boundaries'),('bowl','after_two_sixes')]:
        pid=IDS[r]
        a=v.groupby([pid,trigger]).agg(events=('delivery_id','size'),matches=('match_id','nunique'),total=(r+'_value','sum'))
        if True not in a.index.get_level_values(1):continue
        a=a.xs(True,level=1).join(a.xs(False,level=1).add_suffix('_ordinary'),how='inner')
        pre=a.total_ordinary/a.events_ordinary
        a['value_per_event']=a.total/a.events
        a['shrunken_difference']=(a.total-a.events*pre)/(a.events+50)
        a['trigger']=trigger;a['role']=r
        rows.append(a.reset_index().rename(columns={pid:'player_id'}))
    save(pd.concat(rows,ignore_index=True),'setback_players')
    subsets=[]
    for r,trig in [('bat','after_two_dots'),('bat','collapse_pre'),('bowl','after_two_boundaries'),('bowl','after_two_sixes')]:
        for flag,s in v.groupby(trig):
            subsets.append(dict(role=r,trigger=trig,on=flag,events=len(s),matches=s.match_id.nunique(),runs=s[r+'_runs'].mean(),
                predicted=s[r+'_expected_runs'].mean(), wicket=s[WICKET[r]].mean(),expected_wicket=s[r+'_expected_wicket'].mean(),
                boundary=(s.is_four|s.is_six).mean(),legal_dot=((s.team_runs_event==0)&s.legal_event.eq(1)&~s.is_wicket).mean()))
    save(pd.DataFrame(subsets),'setback_summary')


def write_report(metrics,summary,event,prices,coverage):
    selected=metrics[(metrics['sample']=='all') & (metrics.test_start=='2026-01-01')]
    ablation=summary[(summary['sample']=='all') & (summary.period=='2026 reserved')]
    def effect(r,model):
        a=ablation[(ablation.role==r)&(ablation.model==model)].iloc[0]
        return f"{a.run_delta:+.5f} [{a.run_lo:+.5f}, {a.run_hi:+.5f}]"
    def rates(r):
        a=selected[(selected.role==r)&selected.model.eq('opposition')].iloc[0]
        return f"{100*a.expected_wickets:.2f}% predicted, {100*a.observed_wickets:.2f}% observed"
    p=prices[prices.test_start.eq('2026-01-01')].iloc[0]
    cov=coverage[(coverage.year==2026)&(coverage.role=='bat')].iloc[0]
    short=pd.read_csv(OUT/'response_example_short.csv').iloc[-1]
    across=pd.read_csv(OUT/'response_example_across.csv').iloc[0]
    txt=f"""# Estimating performance on the next recorded delivery

Five expanding forecasts start in October 2022 and January of 2023, 2024, 2025 and 2026. Outcomes and coefficient estimates for each forecast use only matches before its start. Features can update from completed earlier dates inside the forecast window. The 2023–25 windows are development checks. The reserved 2026 window contains {int(cov.events):,} recorded events in {int(selected.matches.iloc[0])} matches. Earlier project reports already examined this season. It is a forward test of newly fitted coefficients, with no claim of a previously unseen competition dataset.

## Opportunity and expected performance

A ridge regression (penalty 200) estimates runs. A logistic regression (C = 0.02) estimates the probability of striker dismissal or a bowler-credited wicket. Both use a training-fold intercept and standardized features. These fixed penalties regularize small interactions and were never tuned on 2026 results. The unconditional mean, match state, role, opposition, individual ability and memory models were fitted separately. The role block adds arrival order, time at the crease, spell stage, prior-observed handedness and bowling style, phase interactions, historical venue scoring and a player's earlier phase share. For two simultaneous new arrivals, position is the mean of their two entry slots. No final batting card is used.

Opponent estimates use performance relative to historical competition × innings × phase norms. The cell mean has 60 pseudo-events at 1.25 runs and 0.05 wickets. Individual, team and venue averages shrink toward zero contextual residual with 100, 500 and 500 events respectively. All current-date fixtures are withheld together. Own ability, recent form and setback history enter forecasting experiments only. The ranking innovations compare performance against the historical role-and-opponent reference. This preserves a strong player's contextual advantage.

The signature of role, expressed as change in held-out match-mean run MSE with paired-match 95% bootstrap intervals, was {effect('bat','role')} for batting and {effect('bowl','role')} for bowling. Adding the historical opponent changed MSE by {effect('bat','opposition')} and {effect('bowl','opposition')}. Calibration in the large for this model was {rates('bat')} among strikers and {rates('bowl')} for bowler wickets. The evidence files contain decile calibration, Brier score, log loss, and separate competition, phase and innings results.

## From outcomes to runs of value

Let r and w denote observed runs and the attributed wicket indicator. Let μ and p denote the forecasts per recorded event. A legal ball receives exposure h = 1 / P(legal), where P(legal) is estimated within prior competition × phase × innings, with 100 events of shrinkage toward the earlier aggregate. An illegal delivery receives h = 0. Batting value is r − hμ + V(hp − w). Bowling reverses the sign, using bowler-conceded runs and credited wickets. This charges wide and no-ball runs to the bowler even when the delivery uses no legal ball. A stumping on a wide can still count. Non-striker run-outs produce no striker dismissal and no bowler wicket. Ambiguous secondary extras and unresolved dismissals are excluded. Batting measures striker contribution and does not charge a dismissed non-striker to the striker. Impact substitutions and absent nonparticipants are never interpreted as mental states.

The wicket price V is an estimate of future scoring resources. For each wicket j up to the number remaining, a nonnegative regression adds linear, square-root and quadratic functions of balls remaining divided by 120. It fits the subsequent runs of completed earlier first innings, weighting each innings equally, with a smoothness penalty across adjacent wickets. V is the difference between the fitted surface with w and with w−1 wickets in hand after a hypothetical legal ball. We use one pre-specified legal-ball transition for all event types so the current result cannot set its own price. A wicket on the final legal ball has zero future run cost. The same surface is carried to chases, where target truncation would bias a direct future-runs fit. This is a resource estimate and does not quantify the specific match-winning cost of losing the final wicket in a chase.

Across scored events the median price was {event.wicket_price.median():.2f} runs (middle half {event.wicket_price.quantile(.25):.2f}–{event.wicket_price.quantile(.75):.2f}). In 2026 the first-innings future-runs RMSE was {p.heldout_future_runs_rmse:.2f} against {p.constant_run_rate_rmse:.2f} for a constant historical run rate multiplied by balls remaining. The validation checks the surface as a whole. The marginal effect of a wicket is observational and has no experimental validation.

## Memory and opportunity

A dot sequence counts two earlier legal, wicketless team dots on that striker's own recorded events. Boundary and six sequences use the bowler's two earlier recorded deliveries that innings, which can span overs. Collapse uses three wickets in the previous twelve legal balls; an additional indicator tests two. The momentum block includes lagged team acceleration, batter acceleration, spell economy and change in required rate. Three poor performance dates are consecutive signed run-tempo residuals below the player's prior shrunk career level after three previous appearances. Performance divides residual runs by events plus six to temper short appearances. Recent trend is the mean over the previous five observed performance dates. Historical rebound averages prior responses following a below-baseline appearance, shrunk by eight opportunities. These proxies describe scoring tempo independently of dismissal risk. The performance history covers valid standard matches in the supplied event file.

The opportunity experiment uses days since the previous recorded XI appearance, previous 90-day appearance dates and presence in the team's last five supplied fixtures. A usable team record has 11–12 distinct players marked as played. All five prior rosters must be usable to calculate the share. Missingness has explicit indicators. The 2026 batter sample has usable five-fixture coverage on {int(cov.five_prior_rosters):,} of {int(cov.events):,} events. Coverage does not establish contract eligibility, injuries, omissions from the source or the reason for any absence.

All eight memory blocks together changed held-out match-mean run MSE by {effect('bat','all_memory')} for batters and {effect('bowl','all_memory')} for bowlers. Allowing the static pressure score to interact with trend, dot sequences, boundary sequences, collapse and slump changed MSE by {effect('bat','interactions')} and {effect('bowl','interactions')} relative to additive memory. Each interval resamples paired complete matches 1,500 times. Individual tests are exploratory and intervals are unadjusted for multiple comparisons. Coefficients and all single-block ablations are available, including high-pressure-only analyses with prior training-date thresholds. No memory block earns a bonus in the rankings.

## Two recorded sequences

On {short.match_date_parsed}, {short.player_name} had just conceded consecutive sixes in match {short.match_id}, innings {short.innings}, over field {short.overid}. At delivery {short.delivery_id} the earlier score was {short.score_pre} with {short.wickets_remaining_pre} wickets in hand. The memory-and-interaction model predicted {short.expected_runs:.3f} conceded runs and {100*short.wicket_prob:.2f}% wicket probability. Turning off the two boundary flags and their pressure interaction within that same model gives {short.flag_off_expected_runs:.3f} and {100*short.flag_off_wicket_prob:.2f}%. The separate no-memory ability model estimated {short.baseline_runs:.3f} and {100*short.baseline_wicket:.2f}%. The observed outcome was {short.bowl_runs:.0f} run(s). The flag-off comparison isolates fitted terms while holding the observed match state fixed.

On {across.match_date_parsed}, {across.player_name} entered match {across.match_id} after four below-baseline scoring-tempo dates. At delivery {across.delivery_id} his recent five-date proxy was {across.form_trend:+.3f}. The full forecast was {across.expected_runs:.3f} runs and {100*across.wicket_prob:.2f}% dismissal probability. Holding the rest of his history fixed and switching off the three-poor flag plus interaction gives {across.flag_off_expected_runs:.3f} and {100*across.flag_off_wicket_prob:.2f}%. The next event yielded {across.bat_runs:.0f} runs. The chronological history table identifies the matches and shows how the proxy evolved. Both examples are the first reserved high-pressure event meeting the predeclared trigger, ordered by date, match, innings and event. Neither establishes a psychological response.

Reproduce with `python -m final_strategy.response_model` after building Phase 1. Feature names and fit windows are printed and saved in `response_training.log`. The coefficient table includes training means, scales and both intercepts to reconstruct forecasts. The wicket-surface table contains its fitted coefficients. Event innovations and ablation, calibration, segment, history-coverage and descriptive recovery tables are adjacent to this document.
"""
    (OUT/'response_methods.md').write_text(txt,encoding='utf-8')


if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
