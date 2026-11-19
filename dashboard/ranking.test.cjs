const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const rank = vm.runInNewContext(fs.readFileSync(path.join(__dirname, 'ranking.js'), 'utf8') + '\nrankingRows');
const players = [
  {role:'bat', player_id:10, player_name:'Same', rank:1, rating:1, eligible:true, supported:true, absolute:2, uplift:1, positive_own_share:.7},
  {role:'bat', player_id:11, player_name:'Same', rank:null, rating:4, eligible:false, supported:true, absolute:5, uplift:3, positive_own_share:.5},
  {role:'bowl', player_id:10, rank:1, rating:2, eligible:true},
];
test('official and explorer use saved role-specific identity and ranks', () => {
  assert.deepEqual(rank(players,'bat','approved',1,0).map(x=>x.p.player_id), [10]);
  const rows = rank(players,'bat','all',1,0);
  assert.deepEqual(rows.map(x=>x.rank), [1,'Unranked']);
  assert.deepEqual(rows.map(x=>x.p.player_id), [10,11]);
  assert.equal(rank(players,'bowl','approved',0,50)[0].score,2);
});
test('experimental threshold is strict and leaves official rank intact', () => {
  assert.equal(rank(players,'bat','scenario',1,50).length,1);
  const first=rank(players,'bat','scenario',1,49)[0];
  assert.equal(first.p.player_id,11);
  assert.equal(first.score,3);
  assert.equal(first.p.rank,null);
});
