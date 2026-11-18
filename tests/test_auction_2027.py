from pathlib import Path

import pytest

from final_strategy.auction_2027 import OUT, ROSTER, read


def test_retention_identity_and_exclusion():
    roster = {r['official_name'] for r in read(ROSTER)}
    audit = read(OUT/'final_availability_audit.csv')
    assert len(roster) == 184 and len({r['Official ID'] for r in audit}) == len(audit) == 38
    siraj, = [r for r in audit if r['Official ID']=='5944']
    assert (siraj['Nationality'],siraj['Retaining franchise']) == ('India','Gujarat Titans')
    for r in audit:
        name = 'Mohammad Siraj' if r['Official ID']=='5944' else r['Player']
        assert (name in roster) == r['IPL 2027 status'].startswith('RETAINED')
        if name in roster:
            assert r['Decision']=='Excluded'
        assert r['Source']


@pytest.mark.parametrize('role,label,expected', [('bat','batters',1),('bowl','bowlers',3)])
def test_candidate_integrity_and_quota(role,label,expected):
    original = read(OUT/'auction_2027_filtered_rankings.csv')
    audit = {r['Official ID']:r for r in read(OUT/'final_availability_audit.csv')}
    source = [r for r in original if r['role']==role and audit[r['player_id']]['Decision']=='Included']
    result = read(OUT/f'final_auction_filtered_{label}.csv')
    assert len(result)==len({r['Official ID'] for r in result})==min(10,len(source))==expected
    assert [r['Official ID'] for r in result]==[r['player_id'] for r in source]
    for r,s in zip(result,source):
        assert r['Player']==audit[r['Official ID']]['Player']==s['player_name']
        assert r['Runs per six']==s['rating'] and r['Original rank']==s['historical_rank']
        assert r['IPL 2027 availability']=='Released; registration unverified'
        assert audit[r['Official ID']]['Source']=='L'
        assert r['Player'] in (OUT/'final_results_for_report.md').read_text()
