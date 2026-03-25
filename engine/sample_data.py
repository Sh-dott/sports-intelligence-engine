"""
Sample Data Generator
Creates realistic match event data for testing and demonstration.
"""

import random
import pandas as pd
import numpy as np


def generate_football_match(seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic 90-minute football match between two teams.
    Includes deliberate patterns for the detection engine to find:
    - A momentum shift around minute 35
    - A defensive collapse for Team B around minute 55-60
    - Late-game surge by Team B
    - Critical mistakes leading to goals
    """
    random.seed(seed)
    np.random.seed(seed)

    teams = ["Arsenal", "Chelsea"]
    players = {
        "Arsenal": ["Saka", "Odegaard", "Rice", "Saliba", "Havertz",
                     "Ramsdale", "White", "Gabriel", "Zinchenko", "Trossard", "Martinelli"],
        "Chelsea": ["Palmer", "Jackson", "Caicedo", "Colwill", "Sanchez",
                     "James", "Silva", "Mudryk", "Sterling", "Gallagher", "Nkunku"],
    }

    events = []
    minute = 0.0

    def add_event(t, team, player, event_type, loc_x=None, loc_y=None, detail=""):
        events.append({
            "timestamp": round(t, 2),
            "team": team,
            "player": player,
            "event_type": event_type,
            "location_x": loc_x if loc_x is not None else random.uniform(0, 100),
            "location_y": loc_y if loc_y is not None else random.uniform(0, 100),
            "detail": detail,
            "period": 1 if t <= 45 else 2,
            "match_id": "EPL-2024-001",
        })

    def random_player(team):
        return random.choice(players[team])

    # === FIRST HALF: Arsenal dominant early ===

    # Arsenal early pressure (0-20 min)
    for m in np.arange(1, 20, 0.3 + random.random() * 0.5):
        team = "Arsenal" if random.random() < 0.65 else "Chelsea"
        p = random_player(team)
        if team == "Arsenal":
            evt = random.choices(
                ["pass", "dribble", "cross", "shot", "shot_on_target", "corner"],
                weights=[40, 15, 10, 8, 5, 5]
            )[0]
            add_event(m, team, p, evt, loc_x=random.uniform(50, 95))
        else:
            evt = random.choices(
                ["pass", "clearance", "tackle", "interception", "foul"],
                weights=[30, 15, 15, 15, 10]
            )[0]
            add_event(m, team, p, evt, loc_x=random.uniform(10, 50))

    # Arsenal goal at 12 min — Saka scores after pressure
    add_event(12.0, "Arsenal", "Odegaard", "pass", loc_x=70, detail="through ball")
    add_event(12.2, "Arsenal", "Saka", "shot_on_target", loc_x=88, detail="right foot")
    add_event(12.3, "Arsenal", "Saka", "goal", loc_x=92, detail="bottom left corner")

    # Continued Arsenal pressure 20-33
    for m in np.arange(20, 33, 0.4 + random.random() * 0.6):
        team = "Arsenal" if random.random() < 0.6 else "Chelsea"
        p = random_player(team)
        evt = random.choices(
            ["pass", "dribble", "tackle", "shot", "foul", "corner", "cross"],
            weights=[35, 10, 12, 8, 8, 5, 7]
        )[0]
        add_event(m, team, p, evt)

    # === MOMENTUM SHIFT at ~35 min ===
    # Chelsea critical mistake: Rice loses ball in midfield
    add_event(34.5, "Arsenal", "Rice", "pass", loc_x=55, detail="misplaced pass")
    add_event(34.7, "Chelsea", "Caicedo", "interception", loc_x=50, detail="intercepted")
    add_event(35.0, "Chelsea", "Palmer", "dribble", loc_x=65)
    add_event(35.3, "Chelsea", "Palmer", "shot_on_target", loc_x=82)
    add_event(35.5, "Chelsea", "Palmer", "goal", loc_x=90, detail="equalizer, curling shot")

    # Chelsea takes over 35-45
    for m in np.arange(36, 45, 0.3 + random.random() * 0.4):
        team = "Chelsea" if random.random() < 0.7 else "Arsenal"
        p = random_player(team)
        if team == "Chelsea":
            evt = random.choices(
                ["pass", "shot", "shot_on_target", "cross", "dribble", "corner", "free_kick"],
                weights=[30, 10, 8, 10, 12, 5, 5]
            )[0]
            add_event(m, team, p, evt, loc_x=random.uniform(55, 95))
        else:
            evt = random.choices(
                ["pass", "clearance", "foul", "tackle"],
                weights=[30, 20, 15, 15]
            )[0]
            add_event(m, team, p, evt, loc_x=random.uniform(10, 45))

    # Chelsea second goal at 43
    add_event(43.0, "Chelsea", "Sterling", "cross", loc_x=85)
    add_event(43.2, "Chelsea", "Jackson", "header", loc_x=90)
    add_event(43.3, "Chelsea", "Jackson", "goal", loc_x=92, detail="header from cross")

    # === SECOND HALF ===

    # Arsenal pushing but wasteful 45-55
    for m in np.arange(46, 55, 0.35 + random.random() * 0.5):
        team = "Arsenal" if random.random() < 0.6 else "Chelsea"
        p = random_player(team)
        if team == "Arsenal":
            evt = random.choices(
                ["pass", "shot", "cross", "dribble", "corner", "free_kick"],
                weights=[30, 15, 10, 10, 5, 5]
            )[0]
            add_event(m, team, p, evt, loc_x=random.uniform(55, 90))
        else:
            evt = random.choices(
                ["pass", "tackle", "clearance", "interception"],
                weights=[30, 20, 15, 15]
            )[0]
            add_event(m, team, p, evt)

    # === ARSENAL DEFENSIVE COLLAPSE 55-60 ===
    # Chelsea scores twice in quick succession

    # Critical mistake by Saliba
    add_event(55.0, "Arsenal", "Saliba", "pass", loc_x=35, detail="poor back pass")
    add_event(55.2, "Chelsea", "Nkunku", "interception", loc_x=40, detail="pounced on error")
    add_event(55.5, "Chelsea", "Nkunku", "shot_on_target", loc_x=85)
    add_event(55.7, "Chelsea", "Nkunku", "goal", loc_x=90, detail="one-on-one finish")

    # Arsenal foul in frustration
    add_event(57.0, "Arsenal", "Rice", "foul", loc_x=45, detail="frustrated challenge")
    add_event(57.1, "Arsenal", "Rice", "yellow_card", loc_x=45)

    # Chelsea fourth goal
    add_event(59.0, "Chelsea", "Palmer", "free_kick", loc_x=75)
    add_event(59.3, "Chelsea", "Palmer", "shot_on_target", loc_x=82)
    add_event(59.5, "Chelsea", "Palmer", "goal", loc_x=88, detail="free kick into top corner")

    # Midfield battle 60-75
    for m in np.arange(61, 75, 0.4 + random.random() * 0.5):
        team = random.choice(teams)
        p = random_player(team)
        evt = random.choices(
            ["pass", "tackle", "dribble", "foul", "shot", "clearance", "interception"],
            weights=[30, 12, 10, 10, 8, 8, 7]
        )[0]
        add_event(m, team, p, evt)

    # Arsenal substitution and tactical change
    add_event(70.0, "Arsenal", "Trossard", "substitution", detail="Trossard off")
    add_event(70.0, "Arsenal", "Martinelli", "substitution", detail="Martinelli on")

    # === ARSENAL LATE GAME SURGE 75-90 ===
    for m in np.arange(75, 88, 0.25 + random.random() * 0.3):
        team = "Arsenal" if random.random() < 0.75 else "Chelsea"
        p = random_player(team)
        if team == "Arsenal":
            evt = random.choices(
                ["pass", "shot", "shot_on_target", "cross", "corner", "dribble", "free_kick"],
                weights=[25, 15, 10, 12, 8, 10, 5]
            )[0]
            add_event(m, team, p, evt, loc_x=random.uniform(60, 98))
        else:
            evt = random.choices(
                ["clearance", "tackle", "foul", "pass"],
                weights=[25, 20, 15, 20]
            )[0]
            add_event(m, team, p, evt, loc_x=random.uniform(5, 40))

    # Arsenal consolation goal at 82
    add_event(82.0, "Arsenal", "Martinelli", "dribble", loc_x=75)
    add_event(82.3, "Arsenal", "Martinelli", "shot_on_target", loc_x=88)
    add_event(82.5, "Arsenal", "Martinelli", "goal", loc_x=92, detail="solo run and finish")

    # Arsenal second consolation at 87
    add_event(87.0, "Arsenal", "Havertz", "header", loc_x=90)
    add_event(87.2, "Arsenal", "Havertz", "goal", loc_x=92, detail="header from corner")

    # Final whistle
    add_event(90.0, "Arsenal", "", "fulltime")

    df = pd.DataFrame(events)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def generate_basketball_match(seed: int = 42) -> pd.DataFrame:
    """Generate a realistic basketball game with patterns to detect."""
    random.seed(seed)
    np.random.seed(seed)

    teams = ["Lakers", "Celtics"]
    players = {
        "Lakers": ["James", "Davis", "Reaves", "Russell", "Hachimura"],
        "Celtics": ["Tatum", "Brown", "White", "Holiday", "Porzingis"],
    }

    events = []
    score = {"Lakers": 0, "Celtics": 0}

    def add_event(t, team, player, event_type, detail=""):
        events.append({
            "timestamp": round(t, 2),
            "team": team,
            "player": player,
            "event_type": event_type,
            "location_x": random.uniform(0, 94),
            "location_y": random.uniform(0, 50),
            "detail": detail,
            "period": 1 if t <= 12 else (2 if t <= 24 else (3 if t <= 36 else 4)),
            "match_id": "NBA-2024-001",
        })

    def random_player(team):
        return random.choice(players[team])

    minute = 0.5

    # Q1: Back and forth, Lakers slightly better
    while minute < 12:
        team = "Lakers" if random.random() < 0.55 else "Celtics"
        p = random_player(team)
        r = random.random()

        if r < 0.25:
            add_event(minute, team, p, "field_goal")
            score[team] += 2
        elif r < 0.35:
            add_event(minute, team, p, "three_pointer")
            score[team] += 3
        elif r < 0.45:
            add_event(minute, team, p, "field_goal_miss")
        elif r < 0.55:
            other = [t for t in teams if t != team][0]
            add_event(minute, other, random_player(other), "rebound")
        elif r < 0.65:
            add_event(minute, team, p, "assist")
        elif r < 0.72:
            add_event(minute, team, p, "turnover")
        elif r < 0.78:
            other = [t for t in teams if t != team][0]
            add_event(minute, other, random_player(other), "steal")
        elif r < 0.85:
            add_event(minute, team, p, "free_throw")
            score[team] += 1
        else:
            add_event(minute, team, p, "personal_foul")

        minute += random.uniform(0.2, 0.6)

    # Q2: Celtics scoring run (8-0 run)
    for i in range(4):
        m = 13 + i * 0.8
        p = random.choice(["Tatum", "Brown"])
        if random.random() < 0.4:
            add_event(m, "Celtics", p, "three_pointer")
            score["Celtics"] += 3
        else:
            add_event(m, "Celtics", p, "field_goal")
            score["Celtics"] += 2

    # Lakers turnovers during the run
    for i in range(3):
        m = 13.3 + i * 0.8
        add_event(m, "Lakers", random_player("Lakers"), "turnover")

    # Rest of Q2
    minute = 17
    while minute < 24:
        team = random.choice(teams)
        p = random_player(team)
        r = random.random()
        if r < 0.22:
            add_event(minute, team, p, "field_goal")
            score[team] += 2
        elif r < 0.32:
            add_event(minute, team, p, "three_pointer")
            score[team] += 3
        elif r < 0.45:
            add_event(minute, team, p, "field_goal_miss")
        elif r < 0.55:
            add_event(minute, team, p, "turnover")
        elif r < 0.65:
            add_event(minute, team, p, "rebound")
        elif r < 0.75:
            add_event(minute, team, p, "personal_foul")
        else:
            add_event(minute, team, p, "free_throw")
            score[team] += 1
        minute += random.uniform(0.2, 0.5)

    # Q3: Lakers comeback with James dominance
    minute = 25
    while minute < 36:
        team = "Lakers" if random.random() < 0.65 else "Celtics"
        p = random_player(team)
        if team == "Lakers":
            p = "James" if random.random() < 0.45 else p
            r = random.random()
            if r < 0.3:
                add_event(minute, team, p, "field_goal")
                score[team] += 2
            elif r < 0.42:
                add_event(minute, team, p, "three_pointer")
                score[team] += 3
            elif r < 0.55:
                add_event(minute, team, p, "assist")
            elif r < 0.65:
                add_event(minute, team, p, "rebound")
            else:
                add_event(minute, team, p, "field_goal_miss")
        else:
            r = random.random()
            if r < 0.2:
                add_event(minute, team, p, "field_goal")
                score[team] += 2
            elif r < 0.35:
                add_event(minute, team, p, "field_goal_miss")
            elif r < 0.5:
                add_event(minute, team, p, "turnover")
            else:
                add_event(minute, team, p, "personal_foul")
        minute += random.uniform(0.2, 0.5)

    # Q4: Tight finish
    minute = 37
    while minute < 48:
        team = random.choice(teams)
        p = random_player(team)
        r = random.random()
        if r < 0.2:
            add_event(minute, team, p, "field_goal")
            score[team] += 2
        elif r < 0.3:
            add_event(minute, team, p, "three_pointer")
            score[team] += 3
        elif r < 0.42:
            add_event(minute, team, p, "field_goal_miss")
        elif r < 0.52:
            add_event(minute, team, p, "free_throw")
            score[team] += 1
        elif r < 0.62:
            add_event(minute, team, p, "rebound")
        elif r < 0.72:
            add_event(minute, team, p, "turnover")
        elif r < 0.8:
            add_event(minute, team, p, "steal")
        elif r < 0.88:
            add_event(minute, team, p, "block")
        else:
            add_event(minute, team, p, "personal_foul")
        minute += random.uniform(0.15, 0.45)

    df = pd.DataFrame(events)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df
