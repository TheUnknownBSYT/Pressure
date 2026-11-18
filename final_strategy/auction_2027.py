"""Screen the frozen eligible pool using the IPL's 18 November 2026 contract snapshot."""
import csv
from pathlib import Path

OUT = Path("submission")
ROSTER = Path(__file__).with_name("retentions_2027.csv")
RELEASE = {"3297": ("Shimron Hetmyer", "Finisher"), "3653": ("Avesh Khan", "Indian pace"),
           "8661": ("Matheesha Pathirana", "Death bowling"), "6523": ("Blessing Muzarabani", "Seam bowling")}
RETAIN = "https://www.iplt20.com/news/article/tata-ipl-2027-player-retentions-announced"
RELEASE_URL = "https://www.iplt20.com/cms/files/jot2qw8s50z1/3gFAJ4z1USxcXYhaZVl65L/2a463a9c02a5bb7fd12fa3fbe91cc299/TATA_IPL_2027-_Release_Player_List.pdf"
PCB = "https://www.pcb.com.pk/press-release-detail/pakistan-announce-squad-for-icc-men-s-t20-world-cup-2026-.html"
POLICY = "https://www.cricbuzz.com/cricket-news/140773/mohammad-amir-rules-out-entering-ipl-auction-cricbuzzcom"
SIRAJ = "https://www.icc-cricket.com/rankings/65799/mohammed-siraj"
RETIRE = "https://indianexpress.com/article/sports/cricket/kolkata-knight-riders-kkr-retire-andre-russell-no-12-jersey-ipl-2026-10600196/"
STATUS = {"RR":"RETAINED_BY_RR", "Other":"RETAINED_BY_OTHER_IPL_TEAM",
          "Released":"RELEASED_OR_UNRETAINED_POTENTIAL_CANDIDATE",
          "Out":"NOT_ELIGIBLE_FOR_IPL_2027","Unknown":"AVAILABILITY_UNVERIFIED"}


def read(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def save(name, headers, rows):
    with (OUT/name).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def table(name):
    with (OUT/name).open() as f:
        headers, *rows = csv.reader(f)
    return "\n".join("| " + " | ".join(r) + " |" for r in [headers, ["---"]*len(headers), *rows])


def run():
    roster = {r["official_name"]: r["franchise"] for r in read(ROSTER)}
    assert len(roster) == 184
    mapping = read("competition_team_player_mapping.csv")
    identities = {(r["player_id"], r["player_name"]) for r in mapping}
    audit, tops = {}, {}
    headers = ["Rank","Player","Official ID","Original rank","Runs per six","Pressure matches","Weighted success %","IPL 2027 availability","Potential RR role"]
    for role, label in [("bat","batters"),("bowl","bowlers")]:
        pool = read(f"outputs/auction_pool_{role}.csv")
        ranks = [int(r["rank"]) for r in pool]
        assert ranks == list(range(1,len(pool)+1)) and len({r["player_id"] for r in pool}) == len(pool)
        tops[role] = [r["player_name"] for r in pool[:10]]
        selected = []
        for r in pool:
            pid, name = r["player_id"], r["player_name"]
            if {p for p,n in identities if n==name} != {pid} or {n for p,n in identities if p==pid} != {name}:
                raise ValueError(f'Unresolved identity: {pid} {name}')
            official = "Mohammad Siraj" if (pid,name)==("5944","Mohammed Siraj") else name
            team = roster.get(official,"")
            nation, source, status = "Unknown","R/L","Unknown"
            if team:
                status,source = ("RR" if team=="Rajasthan Royals" else "Other"),"R"
            elif pid in RELEASE:
                assert RELEASE[pid][0]==name
                status,source = "Released","L"
                selected.append([len(selected)+1,name,pid,r["rank"],r["rating"],r["pressure_matches"],r["weighted_success_pct"],"Released; registration unverified",RELEASE[pid][1]])
            elif pid in {"5979","6355"}:
                nation,status,source = "Pakistan","Out","P/E"
            elif pid=="1318":
                status,source = "Out","T"
            if pid=="5944":
                nation,source = "India","R/I"
            audit[pid] = [name,nation,pid,STATUS[status],team or "—","Included" if status=="Released" else "Excluded",source]
        save(f"final_auction_filtered_{label}.csv",headers,selected)
    save("final_unrestricted_rankings.csv",["Rank","Batter","Bowler"],[(i+1,*pair) for i,pair in enumerate(zip(tops["bat"],tops["bowl"]))])
    save("final_availability_audit.csv",["Player","Nationality","Official ID","IPL 2027 status","Retaining franchise","Decision","Source"],audit.values())
    report()


def report():
    sections = ["# Round 1 results and IPL 2027 auction screen", "## A. Unrestricted statistical rankings", table("final_unrestricted_rankings.csv")]
    for title,role in [("B. Auction-filtered batting","batters"),("C. Auction-filtered bowling","bowlers")]:
        rows = read(OUT/f"final_auction_filtered_{role}.csv")
        lines = ["| Rank | Player | Original rank | Runs per six | Pressure matches | Weighted success | IPL 2027 availability | Potential RR role |","|---|---|---|---|---|---|---|---|"]
        for r in rows:
            lines.append("| {} | {} | {} | {:+.3f} | {} | {:.2f}% | Released, registration unverified | {} |".format(r["Rank"],r["Player"],r["Original rank"],float(r["Runs per six"]),r["Pressure matches"],float(r["Weighted success %"]),r["Potential RR role"]))
        sections.extend(["## "+title,"\n".join(lines)])
    sections.extend(["## D. Availability audit","Unknown nationality means citizenship was not independently verified. P identifies Pakistan cricket representation. Out includes IPL retirement. R/L means neither official list establishes availability. All 38 distinct players are shown.","| Player | Nationality | ID | Status | Retained by | Decision | Source |\n|---|---|---|---|---|---|---|"])
    status_labels = {v:k for k,v in STATUS.items()}
    teams = {"Chennai Super Kings":"CSK","Delhi Capitals":"DC","Gujarat Titans":"GT","Kolkata Knight Riders":"KKR","Lucknow Super Giants":"LSG","Mumbai Indians":"MI","Punjab Kings":"PBKS","Rajasthan Royals":"RR","Royal Challengers Bengaluru":"RCB","Sunrisers Hyderabad":"SRH"}
    for r in read(OUT/"final_availability_audit.csv"):
        sections[-1] += "\n| "+" | ".join([r["Player"],r["Nationality"],r["Official ID"],status_labels[r["IPL 2027 status"]],teams.get(r["Retaining franchise"],"—"),r["Decision"],r["Source"]])+" |"
    sections.append(f"[R · IPL retentions, 18 Nov 2026]({RETAIN}) · [L · IPL releases, 18 Nov 2026]({RELEASE_URL}) · [I · ICC Siraj profile, checked 18 Nov]({SIRAJ}) · [P · PCB squad, 25 Jan 2026]({PCB}) · [E · IPL participation reporting, Nov 2026]({POLICY}) · [T · Russell retirement, 2026]({RETIRE}).")
    targets = """## E. Best actionable RR targets

| Player | Role | Pressure evidence | Why RR should consider them | Important limitation |
|---|---|---|---|---|
| Matheesha Pathirana | Death pace | 159 IPL balls, 62% success overall | Complements Archer and Sandeep late in the innings | Conservative score only +0.016 |
| Avesh Khan | Indian pace | 260 IPL balls across the matched sample | Adds domestic pace depth after Deshpande’s release | Success narrowly clears 50% |
| Shimron Hetmyer | Left-handed finisher | 196 IPL balls, 62% success overall | Buyback option for the finishing role | Conservative score below zero |
| Blessing Muzarabani | Overseas seam | +0.694 per six, 81% success | Alternative overseas pace option | All 94 matched balls came in T20Is |"""
    (OUT/"final_auction_recommendations.md").write_text(targets+"\n\nAll four were released. Registration unverified as of 18 November 2026. [IPL release list]("+RELEASE_URL+")\n")
    sections.extend([targets,"## Interpretation",
    "The Round 1 question asks which players repeatedly performed well under pressure across the supplied IPL and T20I matches. We therefore ranked every player who met the same exposure, ordinary-performance and weighted-majority requirements before considering franchise contracts. This preserves the distinction between historical performance and an auction decision. A retained player can provide strong evidence about reliable pressure performance even when another franchise cannot buy him. Ishan Kishan and Jasprit Bumrah lead their respective unrestricted lists.",
    "We screened all 17 eligible batting entries and 22 bowling entries against the official retention and release lists dated 18 November 2026. We used competition player IDs and reviewed spelling differences, including Gujarat’s listing of Mohammed Siraj as Mohammad Siraj. Only an official release or verified auction listing qualified a player for this shortlist. We retained the existing scores and relative order. One batter and three bowlers qualified. Even allowing every unresolved player would yield only two batters and nine bowlers.",
    "RR has six open squad places, including three overseas slots. Pathirana offers IPL evidence at the death, while Avesh offers a domestic pace option that preserves an overseas slot. Hetmyer is the only released batter who meets the model’s requirements. Muzarabani has the strongest conservative bowling score among the released candidates, although his matched evidence is entirely international. Release establishes a route to the auction. Actual registration and inclusion in the final auction list remain unverified for all four players."])
    (OUT/"final_results_for_report.md").write_text("\n\n".join(sections)+"\n")


if __name__ == "__main__":
    run()
