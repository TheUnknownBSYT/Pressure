function rankingRows(players, role, mode, weight, threshold) {
  const rows = players.filter(p => p.role === role);
  if (mode === 'scenario') {
    return rows.filter(p => p.supported && p.absolute > 0 && p.uplift > 0 &&
                            p.positive_own_share > threshold / 100)
      .map(p => ({p, score: (1 - weight) * p.rating + weight * p.uplift}))
      .sort((a, b) => b.score - a.score || b.p.absolute - a.p.absolute || a.p.player_id - b.p.player_id)
      .map((row, i) => ({...row, rank: i + 1}));
  }
  return rows.filter(p => mode === 'all' || p.eligible)
    .sort((a, b) => Number(b.eligible) - Number(a.eligible) ||
                     (a.eligible ? a.rank - b.rank : b.rating - a.rating || a.player_id - b.player_id))
    .map(p => ({p, rank: p.rank ?? 'Unranked', score: p.rating}));
}
