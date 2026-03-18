"""Battle simulation wrapper for team win probabilities."""

import csv
import math
import random
import time
from pathlib import Path

from onceworld.core.battle_logic import Field, Monster
from onceworld.config.runtime import (
    BATTLE_MONSTERS_CSV,
    BATTLE_SIM_DELTA_TIME,
    BATTLE_SIM_DURATION,
    BATTLE_SIM_TRIALS,
)
from onceworld.core.perf import debug_perf, is_debug


class BattleSimulator:
    def __init__(
        self,
        csv_path=BATTLE_MONSTERS_CSV,
        trials=BATTLE_SIM_TRIALS,
        delta_time=BATTLE_SIM_DELTA_TIME,
        duration=BATTLE_SIM_DURATION,
    ):
        self.trials = int(trials)
        self.delta_time = float(delta_time)
        self.duration = float(duration)
        self.battle_monster_base = self._load_battle_monster_base(csv_path)

    def _load_battle_monster_base(self, csv_path):
        base = {}
        csv_file = Path(csv_path)
        with csv_file.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row["PET_NAME"].strip()
                base[name] = {
                    "NO": int(row["NO"]),
                    "PET_NAME": name,
                    "ELEMENT": row["ELEMENT"].strip(),
                    "TYPE": row["TYPE"].strip(),
                    "RANGE": row["RANGE"].strip(),
                    "VIT": int(row["VIT"]),
                    "SPD": int(row["SPD"]),
                    "ATK": int(row["ATK"]),
                    "INT": int(row["INT"]),
                    "DEF": int(row["DEF"]),
                    "MDEF": int(row["MDEF"]),
                    "LUCK": int(row["LUCK"]),
                    "MOV": int(row["MOV"]),
                }
        return base

    def simulate(self, results):
        t_sim_total = time.time()
        team_map = {"team_a": "A", "team_b": "B", "team_c": "C"}
        positions = {"A": (450.0, 450.0), "B": (550.0, 450.0), "C": (500.0, 500.0)}
        seed_entries = {"A": [], "B": [], "C": []}
        unknown_count = 0
        t_seed_start = time.time()

        for team_key, team_letter in team_map.items():
            for unit in results.get(team_key, {}).get("units", []):
                unit_name = str(unit.get("unit_name", "") or "")
                if not unit_name or unit_name == "unknown":
                    unknown_count += 1
                    continue
                base = self.battle_monster_base.get(unit_name)
                if base is None:
                    unknown_count += 1
                    continue
                level = int(unit.get("level") or 1)
                seed_entries[team_letter].append((base, level))

        teams_with_known = sum(1 for t in ["A", "B", "C"] if len(seed_entries[t]) > 0)
        if teams_with_known < 2:
            debug_perf("battle_sim:seed_setup", t_seed_start)
            debug_perf("battle_sim:total", t_sim_total)
            return {"available": False, "reason": "not_enough_known_units", "unknown_count": unknown_count}

        wins = {"A": 0, "B": 0, "C": 0, "Draw": 0}
        total_steps = 0
        t_trials_start = time.time()

        for _ in range(self.trials):
            teams = {"A": [], "B": [], "C": []}
            all_placed = []

            for team in ["A", "B", "C"]:
                base_x, base_y = positions[team]
                for base, level in seed_entries[team]:
                    monster = Monster(team, dict(base), level=level)

                    placed = False
                    nx = base_x
                    ny = base_y
                    for _k in range(100):
                        nx = float(base_x + random.randint(-350, 350))
                        ny = float(base_y + random.randint(-350, 350))
                        nx = max(50.0, min(950.0, nx))
                        ny = max(50.0, min(950.0, ny))

                        overlap = False
                        for placed_monster in all_placed:
                            if math.hypot(nx - placed_monster.x, ny - placed_monster.y) < 80.0:
                                overlap = True
                                break
                        if not overlap:
                            monster.x = nx
                            monster.y = ny
                            placed = True
                            break

                    if not placed:
                        monster.x = nx
                        monster.y = ny

                    teams[team].append(monster)
                    all_placed.append(monster)

            field = Field(teams)
            while not field.is_finished() and field.time_elapsed < self.duration:
                field.step(self.delta_time)
                total_steps += 1

            winner = field.get_winner()
            if winner in wins:
                wins[winner] += 1
            else:
                wins["Draw"] += 1
        t_trials_end = time.time()

        total = max(1, sum(wins.values()))
        team_probs = {
            "team_a": wins["A"] / total,
            "team_b": wins["B"] / total,
            "team_c": wins["C"] / total,
        }
        best_team = max(team_probs, key=team_probs.get)

        if is_debug():
            seed_ms = (t_trials_start - t_seed_start) * 1000.0
            trials_ms = (t_trials_end - t_trials_start) * 1000.0
            avg_steps = total_steps / max(1, self.trials)
            print(
                f"[perf] battle_sim: seed={seed_ms:.2f} ms "
                f"trials={trials_ms:.2f} ms "
                f"steps={total_steps} avg_steps/trial={avg_steps:.1f}"
            )
        debug_perf("battle_sim:total", t_sim_total)

        return {
            "available": True,
            "team_probs": team_probs,
            "best_team": best_team,
            "best_prob": team_probs[best_team],
            "draw_prob": wins["Draw"] / total,
            "unknown_count": unknown_count,
        }

