"""Experimental features. Every performance-history update is made after the date closes."""
import numpy as np
import pandas as pd

from final_strategy import response_model as response


def competition_family(names):
    result = pd.Series('unknown', index=names.index)
    rules = [
        ('bilateral', r'\b(?:in|v|vs)\b.*\b(?:series|match)\b'),
        ('multi-team series', r'tri[ -]?(?:series|nation)|quadrangular'),
        ('Asia Cup', r'asia cup'), ('Asia Cup qualifying', r'asia cup.*qualif'),
        ('development tournament', r'emerging|rising stars'), ('IPL', r'^IPL\b'),
    ]
    for label, expression in rules:
        result.loc[names.str.contains(expression, case=False, na=False)] = label
    return result


def locations(fixtures):
    """Country only. UK cannot distinguish England from Scotland. IPL has domestic franchises."""
    m = fixtures
    home = m.home_team.replace({'United States Of America': 'United States'})
    away = m.away_team.replace({'United States Of America': 'United States'})
    place = m.venue_country
    undecidable = (m.match_format.ne('T20I') | place.isna() |
                   (place.eq('United Kingdom') & (home.isin(['England', 'Scotland']) | away.isin(['England', 'Scotland']))))
    return np.select([undecidable, home.eq(place), away.eq(place)],
                     ['unknown', 'listed_home_local', 'listed_away_local'], default='neutral_country')


def prior_ratings(d, role, half_life=365):
    """Context residuals versus earlier competition/phase/innings, decayed to the match date."""
    cell = ['competition_type', 'phase_pre', 'innings']
    pid, team = response.IDS[role], response.TEAMS[role]
    a = d[[response.DATE, pid, team, *cell]].copy()
    a['n'], a['r'], a['w'] = 1., d[role+'_runs'], d[response.WICKET[role]]
    prior = response.attach(a, response.date_prior(a, cell, ['n','r','w']), cell+[response.DATE])
    a['r'] -= (prior.hist_r + 75) / (prior.hist_n + 60)
    a['w'] = a.w.astype(float) - (prior.hist_w + 3) / (prior.hist_n + 60)
    weights = np.exp2((a[response.DATE]-a[response.DATE].min()).dt.days / half_life)
    a[['n','r','w']] = a[['n','r','w']].mul(weights, axis=0)
    out = {}
    for key, label, pseudo in [(pid, 'actual', 100), (team, 'team', 500)]:
        hist = response.attach(a, response.date_prior(a, [key], ['n','r','w']), [key,response.DATE])
        for field in ['r','w']:
            out[label+'_'+field] = hist['hist_'+field] / (hist.hist_n + pseudo*weights)
    return pd.DataFrame(out)


def feature_sets(d, x, fixtures):
    meta = fixtures.set_index('match_id')
    venue = d.match_id.map(meta.location)
    bat_listed_home = d.batting_team_id.eq(d.match_id.map(meta.home_team_id))
    local = ((venue.eq('listed_home_local') & bat_listed_home) |
             (venue.eq('listed_away_local') & ~bat_listed_home))
    visiting = ((venue.eq('listed_home_local') & ~bat_listed_home) |
                (venue.eq('listed_away_local') & bat_listed_home))
    comp = ['ipl_'+c for c in ['powerplay','death','chase','resources','demand']]
    base = response.BASE + response.ROLE
    models = {'no_opp': base, 'published': base+response.OPP, 'competition': base+comp,
              'team': base+comp+response.OPP[2:], 'actual': base+comp+response.OPP[:2],
              'both': base+comp+response.OPP, 'decayed': base+comp+['actual_r','actual_w','team_r','team_w'],
              'home': base+response.OPP+['local_batting','visiting_batting','location_known'],
              'form': base+response.OPP+['form_trend','strong_three','poor_three'],
              'scope': base+response.OPP, 'ipl_only':base+response.OPP}
    for role, features in x.items():
        for name in comp:
            features[name] = features.ipl * features[name[4:]]
        features[['local_batting','visiting_batting','location_known']] = pd.DataFrame(
            {'local_batting':local,'visiting_batting':visiting,'location_known':venue.ne('unknown')}).astype(float)
        decayed = prior_ratings(d,response.OTHER[role])
        features[decayed.columns] = decayed
    return models
