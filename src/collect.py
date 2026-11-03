"""Collect NBA data. Run from the repo root: python src/collect.py"""
import time, os
import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv2

SEASONS = ["2023-24", "2024-25"]
OUT = "data/raw"
os.makedirs(OUT, exist_ok=True)


def get_games(season):
    """One row per team per game."""
    gf = leaguegamefinder.LeagueGameFinder(
        season_nullable=season,
        league_id_nullable="00",
        season_type_nullable="Regular Season",
    )
    df = gf.get_data_frames()[0]
    df["SEASON"] = season
    return df


def get_boxscores(game_ids, cache_path):
    """Player-level box scores, one game at a time, with checkpointing."""
    done = set()
    rows = []
    if os.path.exists(cache_path):
        existing = pd.read_parquet(cache_path)
        done = set(existing["GAME_ID"].unique())
        rows = [existing]
        print(f"Resuming — {len(done)} games already collected")

    todo = [g for g in game_ids if g not in done]
    print(f"{len(todo)} games to fetch")

    consecutive_failures = 0
    for i, gid in enumerate(todo):
        try:
            bs = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=gid, timeout=60)
            rows.append(bs.get_data_frames()[0])
            consecutive_failures = 0
        except Exception as e:
            print(f"  failed {gid}: {e}")
            consecutive_failures += 1
            if consecutive_failures >= 5:
                print("  5 failures in a row — pausing 5 minutes")
                time.sleep(300)
                consecutive_failures = 0
        time.sleep(2.0)

        if (i + 1) % 50 == 0:
            pd.concat(rows, ignore_index=True).to_parquet(cache_path, index=False)
            print(f"  {i+1}/{len(todo)} — checkpoint saved")

    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(cache_path, index=False)
    return out


if __name__ == "__main__":
    all_games = []
    for s in SEASONS:
        print(f"Fetching game logs for {s}")
        all_games.append(get_games(s))
        time.sleep(1)

    games = pd.concat(all_games, ignore_index=True)
    games.to_parquet(f"{OUT}/games.parquet", index=False)
    print(f"Saved {len(games):,} team-game rows")

    game_ids = sorted(games["GAME_ID"].unique())
    print(f"{len(game_ids):,} unique games")

    get_boxscores(game_ids, f"{OUT}/boxscores.parquet")
    print("Done")