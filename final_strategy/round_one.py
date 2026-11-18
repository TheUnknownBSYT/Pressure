"""Export approved rankings from locally held pressure labels and response predictions."""
from pathlib import Path

import pandas as pd

from final_strategy.ratings import ROLE_LABEL, evaluate_sample, load_events, matched_groups
from phase2_pipeline.core import identities


def main():
    data = load_events()
    names = identities(Path("."))[0].set_index("player_id").player_name
    end = pd.to_datetime(pd.read_csv("matches.csv", usecols=["match_date"]).match_date).max()
    selected = []
    output = Path("submission")
    output.mkdir(exist_ok=True)
    for role, label in ROLE_LABEL.items():
        groups = matched_groups(data, role)
        scores, _ = evaluate_sample(groups, end)
        pool = scores.loc[scores.eligible].sort_values("rank").reset_index()
        pool["player_name"] = names.loc[pool.player_id].to_numpy()
        pool["role"] = role
        pool["weighted_success_pct"] = 100 * pool.positive_own_share
        assert pool.player_name.notna().all() and len(pool) >= 10
        for col in ["rank", "pressure_matches", "pressure_legal"]:
            pool[col] = pool[col].astype(int)
        pool.to_csv(Path("outputs") / f"auction_pool_{role}.csv", index=False)
        top = pool.head(10)
        top[["rank", "player_name", "player_id"]].to_csv(output/f"final_{label}_top10.csv", index=False)
        selected.append(top)
    columns = ["role", "rank", "player_name", "player_id", "rating", "pressure_matches",
               "pressure_legal", "weighted_success_pct", "uplift"]
    pd.concat(selected)[columns].to_csv(output/"round_one_rankings.csv", index=False)


if __name__ == "__main__":
    main()
