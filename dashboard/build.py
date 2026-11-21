"""Build the offline dashboard using the official export or a fictional demo."""
import argparse
import csv
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
METRICS = "rating absolute ordinary uplift pressure_matches pressure_legal positive_own_share IPL_legal T20I_legal".split()


def read(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def flag(value):
    return {"True": True, "False": False}[value]


def load_players(path):
    players = []
    keys = set()
    for row in read(path):
        player = {key: float(row[key]) for key in METRICS}
        if not all(math.isfinite(v) for v in player.values()):
            raise ValueError("Invalid numeric value in scored-player export")
        role, pid = row["role"], int(row["player_id"])
        if role not in {"bat", "bowl"} or (role, pid) in keys:
            raise ValueError("Invalid or duplicate role/player ID")
        keys.add((role, pid))
        eligible = flag(row["eligible"])
        rank = float(row["rank"]) if row["rank"] else None
        if eligible != (rank is not None) or (rank is not None and not rank.is_integer()):
            raise ValueError("Inconsistent saved rank and eligibility")
        player.update(role=role, player_id=pid, player_name=row["player_name"],
                      rank=int(rank) if eligible else None, eligible=eligible,
                      supported=flag(row["supported"]))
        players.append(player)
    return players


def verify_order(players, submission):
    for role, label in [("bat", "batters"), ("bowl", "bowlers")]:
        ordered = sorted((p for p in players if p["eligible"] and p["role"] == role),
                         key=lambda p: p["rank"])
        if [p["rank"] for p in ordered] != list(range(1, len(ordered)+1)):
            raise ValueError(f"Invalid saved ranks for {label}")
        expected = read(submission/f"final_{label}_top10.csv")
        actual = [(p["rank"], p["player_id"], p["player_name"]) for p in ordered[:10]]
        if len(expected) != 10 or actual != [(int(r["rank"]), int(r["player_id"]), r["player_name"]) for r in expected]:
            raise ValueError(f"Saved ranks disagree with submission for {label}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    path = HERE/"synthetic.csv" if args.demo else ROOT/"outputs/round_one_scored.csv"
    if not path.exists():
        parser.error("Run python -m final_strategy.round_one first, or use --demo.")
    players = load_players(path)
    if not args.demo:
        verify_order(players, ROOT/"submission")
    template = (HERE/"template.html").read_text().replace("__RANKING_JS__", (HERE/"ranking.js").read_text())
    label = "Synthetic demo · fictional players" if args.demo else "Round 1"
    page = template.replace("__SOURCE__", label).replace("__PLAYERS__", json.dumps(players).replace("<", "\\u003c"))
    destination = ROOT/("Pressure Demo.html" if args.demo else "Pressure Dashboard.html")
    destination.write_text(page)
    print(destination)


if __name__ == "__main__":
    main()
