import json
import math
import random

from onceworld.paths import ATTACK_RANGE_JSON


_ATTACK_RANGE_CACHE = None


def _load_attack_ranges():
    global _ATTACK_RANGE_CACHE
    if _ATTACK_RANGE_CACHE is not None:
        return _ATTACK_RANGE_CACHE

    path = ATTACK_RANGE_JSON
    if not path.exists():
        _ATTACK_RANGE_CACHE = {}
        return _ATTACK_RANGE_CACHE

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        _ATTACK_RANGE_CACHE = {str(k): float(v) for k, v in data.items()}
    except Exception:
        _ATTACK_RANGE_CACHE = {}
    return _ATTACK_RANGE_CACHE


class Monster:
    def __init__(self, t_name, base_data, level=1):
        self.team = t_name
        self.no = int(base_data["NO"])
        self.name = str(base_data["PET_NAME"])
        self.element = str(base_data["ELEMENT"])  # Fire/Water/Wood/Light/Dark
        self.m_type = str(base_data["TYPE"])  # Physical/Magic
        self.range_type = str(base_data.get("RANGE", "Melee"))  # Melee/Ranged
        self.level = int(level)

        lv_multiplier = 1 + (self.level - 1) * 0.1
        self.vit = math.floor(float(base_data["VIT"]) * lv_multiplier)
        self.hp = math.floor(self.vit * 18 + 100)
        self.max_hp = self.hp

        self.spd = math.floor(float(base_data["SPD"]) * lv_multiplier)
        self.atk = math.floor(float(base_data["ATK"]) * lv_multiplier)
        self.int_stat = math.floor(float(base_data["INT"]) * lv_multiplier)

        base_def = math.floor(float(base_data["DEF"]) * lv_multiplier)
        base_mdef = math.floor(float(base_data["MDEF"]) * lv_multiplier)
        self.defense = math.floor(base_def + base_mdef / 10)
        self.mdefense = math.floor(base_mdef + base_def / 10)

        self.luck = math.floor(float(base_data["LUCK"]) * lv_multiplier)
        self.mov = math.floor(float(base_data["MOV"]))

        self.x = 0.0
        self.y = 0.0
        self.cooldown = 0.0
        self.is_dead = False
        self.current_target = None

        self.attack_interval, self.multi_hit, self.ultra_stages = self._calculate_attack_speed()
        self.attack_range = 30.0 if self.range_type == "Melee" else 150.0

        ranges = _load_attack_ranges()
        if self.name in ranges:
            self.attack_range = float(ranges[self.name])

    def _calculate_attack_speed(self):
        points = [
            (0, 1.0),
            (100, 1.5),
            (200, 2.0),
            (300, 2.5),
            (400, 3.0),
            (500, 3.5),
            (600, 4.0),
            (700, 4.5),
            (800, 5.0),
            (3000, 20.0),
        ]

        if self.spd <= 0:
            atk_spd = 1.0
        elif self.spd >= 3000:
            base_hits = 20.0
            extra_multiplier = 1.0
            if self.spd >= 100000:
                extra_multiplier = 5.0
            elif self.spd >= 30000:
                extra_multiplier = 4.0
            elif self.spd >= 10000:
                extra_multiplier = 3.0
            elif self.spd >= 3001:
                extra_multiplier = 2.0
            atk_spd = base_hits * extra_multiplier
        else:
            atk_spd = 1.0
            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]
                if x1 <= self.spd <= x2:
                    atk_spd = y1 + ((self.spd - x1) / (x2 - x1)) * (y2 - y1)
                    break

        if self.spd >= 3000:
            base_hits_per_second = 20
            if self.spd >= 100000:
                ultra_stages = 5
            elif self.spd >= 30000:
                ultra_stages = 4
            elif self.spd >= 10000:
                ultra_stages = 3
            elif self.spd >= 3001:
                ultra_stages = 2
            else:
                ultra_stages = 1
        else:
            base_hits_per_second = max(1, int(atk_spd))
            ultra_stages = 1

        if base_hits_per_second <= 4:
            interval = 1.0 / base_hits_per_second
            base_multi_hit = 1
        else:
            interval = 0.25
            actions_per_second = 4
            base_multi_hit = max(1, round(base_hits_per_second / actions_per_second))

        return interval, base_multi_hit, ultra_stages

    def distance_to(self, other):
        return math.hypot(self.x - other.x, self.y - other.y)

    def move_towards(self, target, delta_time):
        if self.mov == 0:
            speed = 10.0
        else:
            speed = 80.0 * (1 + self.mov * 0.10)

        dist = self.distance_to(target)
        if dist > self.attack_range:
            step = speed * delta_time
            if dist - step < self.attack_range:
                step = dist - self.attack_range

            safe_dist = max(0.001, dist)
            dx = (target.x - self.x) / safe_dist
            dy = (target.y - self.y) / safe_dist
            self.x += dx * step
            self.y += dy * step

    def attack(self, target):
        self.cooldown = self.attack_interval
        logs = []

        hit_chance = 1.0
        luck_ratio = target.luck / max(1, self.luck)
        points = [
            (1.0, 0.99),
            (2.0, 0.434),
            (3.0, 0.046),
            (3.45, 0.023),
            (3.69, 0.021),
            (3.78, 0.019),
            (3.9, 0.0147),
            (4.0, 0.01),
        ]

        if luck_ratio >= 4.0:
            hit_chance = 0.01
        elif luck_ratio <= 1.0:
            hit_chance = 0.99
        else:
            for i in range(len(points) - 1):
                x1, y1 = points[i]
                x2, y2 = points[i + 1]
                if x1 <= luck_ratio <= x2:
                    hit_chance = y1 + ((luck_ratio - x1) / (x2 - x1)) * (y2 - y1)
                    break

        element_mult = 1.0
        attr = self.element
        t_attr = target.element
        if attr == "Fire":
            if t_attr == "Wood":
                element_mult = 1.2
            elif t_attr == "Water":
                element_mult = 0.8
        elif attr == "Water":
            if t_attr == "Fire":
                element_mult = 1.2
            elif t_attr == "Wood":
                element_mult = 0.8
        elif attr == "Wood":
            if t_attr == "Water":
                element_mult = 1.2
            elif t_attr == "Fire":
                element_mult = 0.8
        elif attr == "Light":
            if t_attr == "Dark":
                element_mult = 1.2
            elif t_attr == "Light":
                element_mult = 0.8
        elif attr == "Dark":
            if t_attr == "Light":
                element_mult = 1.2
            elif t_attr == "Dark":
                element_mult = 0.8

        if self.m_type == "Physical":
            base_dmg = (self.atk * 1.75) - target.defense
        else:
            base_dmg = (self.int_stat * 1.75) - target.mdefense

        if base_dmg < 0:
            base_dmg = 0
        base_dmg *= 4.0

        rng_mult = random.uniform(0.9, 1.1)
        crit_mult = 1.0
        is_crit = False
        luck_diff_ratio = abs(self.luck - target.luck) / max(1, max(self.luck, target.luck))
        if luck_diff_ratio <= 0.2 and random.random() < 0.05:
            is_crit = True
            crit_mult = 2.5

        final_dmg_float = base_dmg * element_mult * rng_mult * crit_mult
        dmg = max(1, math.floor(final_dmg_float))

        hits_landed = 0
        for _ in range(self.multi_hit):
            if random.random() < hit_chance:
                hits_landed += 1

        base_damage = dmg * hits_landed
        total_damage = base_damage * self.ultra_stages

        crit_text = "[CRIT] " if is_crit else ""
        if hits_landed == 0:
            hit_pct = hit_chance * 100
            logs.append(f"{self.name} missed {target.name}! (hit chance {hit_pct:.1f}%)")
        elif self.ultra_stages > 1:
            logs.append(
                f"{self.name} dealt {crit_text}{base_damage:,} x {self.ultra_stages} = {total_damage:,} to {target.name}"
            )
        else:
            logs.append(f"{self.name} dealt {crit_text}{total_damage:,} to {target.name}")

        return {"target": target, "total_damage": total_damage, "logs": logs}


class Field:
    def __init__(self, teams):
        self.teams = teams  # {'A': [...], 'B': [...], 'C': [...]}
        self.monsters = []
        for _, mons in self.teams.items():
            for m in mons:
                self.monsters.append(m)
        self.time_elapsed = 0.0

    def step(self, delta_time=0.1):
        logs = []
        if self.is_finished():
            return logs

        self.time_elapsed += delta_time
        living_monsters = [m for m in self.monsters if not m.is_dead]
        living_monsters.sort(key=lambda x: x.spd, reverse=True)

        for m in living_monsters:
            if m.current_target is None or getattr(m.current_target, "is_dead", True):
                nearest_enemy = None
                min_dist = float("inf")
                for enemy in living_monsters:
                    if enemy.team == m.team or m == enemy:
                        continue
                    dist = m.distance_to(enemy)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_enemy = enemy
                m.current_target = nearest_enemy

            if not m.current_target:
                continue

            m.cooldown -= delta_time
            dist = m.distance_to(m.current_target)
            if dist > m.attack_range:
                m.move_towards(m.current_target, delta_time)

        pending_attacks = []
        for m in living_monsters:
            if not m.current_target or getattr(m.current_target, "is_dead", True):
                continue

            dist = m.distance_to(m.current_target)
            t = m.current_target
            if t.mov == 0:
                target_speed = 10.0
            else:
                target_speed = 80.0 * (1 + t.mov * 0.10)
            chase_buffer = target_speed * delta_time

            if dist <= m.attack_range + chase_buffer and m.cooldown <= 0:
                pending_attacks.append(m.attack(m.current_target))

        for result in pending_attacks:
            logs.extend(result["logs"])
            result["target"].hp -= result["total_damage"]

        for m in living_monsters:
            if m.hp <= 0 and not m.is_dead:
                m.is_dead = True
                logs.append(f"{m.name} has fallen!")

        return logs

    def is_finished(self):
        alive_teams = set()
        for m in self.monsters:
            if not m.is_dead:
                alive_teams.add(m.team)
        return len(alive_teams) <= 1

    def _get_team_avg_hp_percentage(self, team_name):
        team_mons = [m for m in self.monsters if m.team == team_name]
        if not team_mons:
            return 0.0
        total_percentage = 0.0
        for m in team_mons:
            if not m.is_dead:
                total_percentage += (m.hp / m.max_hp)
        return total_percentage / len(team_mons) if team_mons else 0.0

    def _get_team_avg_level(self, team_name):
        team_mons = [m for m in self.monsters if m.team == team_name]
        if not team_mons:
            return 0
        return sum(m.level for m in team_mons) / len(team_mons)

    def get_winner(self):
        alive_teams = set(m.team for m in self.monsters if not m.is_dead)

        if len(alive_teams) == 1:
            return list(alive_teams)[0]
        if len(alive_teams) == 0:
            alive_teams = set(self.teams.keys())

        best_hp_pct = -1.0
        hp_candidates = []
        for t in alive_teams:
            thp_pct = self._get_team_avg_hp_percentage(t)
            if thp_pct > best_hp_pct + 0.0001:
                best_hp_pct = thp_pct
                hp_candidates = [t]
            elif abs(thp_pct - best_hp_pct) <= 0.0001:
                hp_candidates.append(t)

        if len(hp_candidates) == 1:
            return hp_candidates[0]

        lowest_lv = float("inf")
        lv_candidates = []
        for t in hp_candidates:
            avg_lv = self._get_team_avg_level(t)
            if avg_lv < lowest_lv:
                lowest_lv = avg_lv
                lv_candidates = [t]
            elif avg_lv == lowest_lv:
                lv_candidates.append(t)

        if lv_candidates:
            return lv_candidates[0]
        return "Draw"
