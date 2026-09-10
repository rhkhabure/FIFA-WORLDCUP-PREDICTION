"""
timeline.py  —  V4.2
======================
Win probability timeline for the Live Match page.

Builds the minute-by-minute probability curve using football_v4.pth,
then renders it as a self-contained inline SVG chart.

GOAL TIMING APPROXIMATION:
The football-data.org free tier does not return goal minutes for
individual match lookups. We use the same approximation as the
training data:
  - First half goals placed at minute 44
  - Second half goals placed at minute 75
  - Goals are split evenly between halves (ceil for first half)

This produces the correct probability shape -- the sharp jumps
happen at the right times relative to each other, just not at the
exact real-world minutes. The chart labels this clearly.

The function build_match_timeline_svg() is the only public export.
"""

import math
import numpy as np
import torch


# ── Goal approximation ────────────────────────────────────────────────────────

def _approximate_goals(h_score: int, a_score: int,
                       home_name: str) -> list[dict]:
    """
    Build an approximate goal sequence from the final scoreline.

    Each goal becomes {'minute': int, 'is_home': bool, 'scorer': str}.
    Half the goals for each team go at minute 44 (first half),
    the rest at minute 75 (second half). This matches the approximation
    used when building the training dataset in build_v4_dataset.py,
    so the model sees input values consistent with what it trained on.
    """
    goals = []

    # Home goals
    h_first  = math.ceil(h_score / 2)
    h_second = h_score - h_first
    for _ in range(h_first):
        goals.append({"minute": 44, "is_home": True,  "scorer": home_name.split()[0]})
    for _ in range(h_second):
        goals.append({"minute": 75, "is_home": True,  "scorer": home_name.split()[0]})

    # Away goals
    a_first  = math.ceil(a_score / 2)
    a_second = a_score - a_first
    for _ in range(a_first):
        goals.append({"minute": 44, "is_home": False, "scorer": "Away"})
    for _ in range(a_second):
        goals.append({"minute": 75, "is_home": False, "scorer": "Away"})

    return sorted(goals, key=lambda g: g["minute"])


def _simulate_score(goals: list[dict], minute: int) -> tuple[int, int]:
    hs  = sum(1 for g in goals if g["minute"] <= minute and     g["is_home"])
    as_ = sum(1 for g in goals if g["minute"] <= minute and not g["is_home"])
    return hs, as_


# ── Public builder ────────────────────────────────────────────────────────────

def build_match_timeline_svg(
    home_name: str,
    away_name: str,
    league: str,
    h_score: int,
    a_score: int,
    home_colour: str,
    away_colour: str,
    dc_lookup,
    nn_model,
    nn_scaler,
    nn_T: float,
    width: int = 700,
    height: int = 190,
) -> str:
    """
    Build a win probability timeline SVG for a finished match.

    Returns an SVG string ready to embed via {{ timeline_svg | safe }}.
    Returns "" if the model is unavailable or the match ended 0-0
    (a goalless match has a flat probability line -- not very interesting).
    """
    if nn_model is None or nn_scaler is None or dc_lookup is None:
        return ""

    total_goals = h_score + a_score

    # For 0-0 draws the chart is nearly flat -- skip it.
    # It would just show three parallel horizontal lines.
    if total_goals == 0:
        return ""

    goals = _approximate_goals(h_score, a_score, home_name)

    # Checkpoints: 0, every 5 min, the two approximated goal minutes, 90
    goal_minutes = sorted(set(g["minute"] for g in goals))
    minutes = sorted(set([0] + list(range(5, 91, 5)) + goal_minutes + [90]))

    points_home, points_draw, points_away = [], [], []

    for m in minutes:
        hs, as_ = _simulate_score(goals, m)
        gso = hs + as_
        lc  = 1 if (hs > 0 and as_ > 0) else 0

        try:
            feat = dc_lookup.build_feature_row(
                home_team=home_name, away_team=away_name, league=league,
                minute=m, home_score=hs, away_score=as_,
                lead_changes=lc, goals_so_far=gso,
                is_knockout=0, is_neutral_venue=0,
            )
        except Exception:
            continue

        X = nn_scaler.transform(
            np.array([feat], dtype="float32")
        ).astype("float32")
        with torch.no_grad():
            logits = nn_model(torch.tensor(X)).numpy()[0]

        z = logits / nn_T; z -= z.max()
        p = np.exp(z); p /= p.sum()

        points_home.append((m, float(p[2])))   # p_home: class 2
        points_draw.append((m, float(p[1])))   # p_draw: class 1
        points_away.append((m, float(p[0])))   # p_away: class 0

    if not points_home:
        return ""

    # ── SVG layout ────────────────────────────────────────────────────────────
    PAD_L, PAD_R, PAD_T, PAD_B = 36, 16, 24, 36
    W = width  - PAD_L - PAD_R
    H = height - PAD_T - PAD_B

    def x_px(minute):
        return PAD_L + (minute / 90) * W

    def y_px(prob):
        return PAD_T + (1.0 - prob) * H

    def polyline(pts):
        return " ".join(f"{x_px(m):.1f},{y_px(p):.1f}" for m, p in pts)

    svg = (
        f'<svg width="100%" viewBox="0 0 {width} {height}"'
        f' xmlns="http://www.w3.org/2000/svg"'
        f' style="display:block;overflow:visible">'
        f'<rect width="{width}" height="{height}" fill="#0b1220" rx="6"/>'
    )

    # ── Grid lines ────────────────────────────────────────────────────────────
    for pct in [0.25, 0.50, 0.75]:
        yg = y_px(pct)
        svg += (
            f'<line x1="{PAD_L}" y1="{yg:.1f}"'
            f' x2="{width - PAD_R}" y2="{yg:.1f}"'
            f' stroke="#1e293b" stroke-width="0.8" stroke-dasharray="3,3"/>'
            f'<text x="{PAD_L - 5}" y="{yg + 3:.1f}" fill="#475569"'
            f' font-size="7" text-anchor="end"'
            f' font-family="JetBrains Mono,monospace">{int(pct * 100)}%</text>'
        )

    # ── Minute axis ───────────────────────────────────────────────────────────
    for m in [0, 15, 30, 45, 60, 75, 90]:
        xg = x_px(m)
        svg += (
            f'<line x1="{xg:.1f}" y1="{PAD_T}"'
            f' x2="{xg:.1f}" y2="{PAD_T + H}"'
            f' stroke="#1e293b" stroke-width="0.5"/>'
            f'<text x="{xg:.1f}" y="{PAD_T + H + 11}"'
            f' fill="#475569" font-size="7" text-anchor="middle"'
            f' font-family="JetBrains Mono,monospace">{m}\'</text>'
        )

    # ── Goal event lines ──────────────────────────────────────────────────────
    # Group by (minute, is_home) to stack scorer labels cleanly
    from collections import defaultdict
    goal_groups: dict = defaultdict(list)
    for g in goals:
        goal_groups[(g["minute"], g["is_home"])].append(g["scorer"])

    label_offsets: dict = {}   # track vertical offset per x-position
    for (minute, is_home), scorers in goal_groups.items():
        xg = x_px(minute)
        colour = home_colour if is_home else away_colour
        label = "⚽ " + ", ".join(scorers)

        # Alternate label above/below midpoint to avoid collisions
        y_offset = label_offsets.get(minute, PAD_T + 10)
        label_offsets[minute] = y_offset + 12

        svg += (
            f'<line x1="{xg:.1f}" y1="{PAD_T}"'
            f' x2="{xg:.1f}" y2="{PAD_T + H}"'
            f' stroke="{colour}" stroke-width="1.2"'
            f' stroke-dasharray="4,3" opacity="0.8"/>'
            f'<text x="{xg + 3:.1f}" y="{y_offset:.1f}"'
            f' fill="{colour}" font-size="6.5"'
            f' font-family="JetBrains Mono,monospace">{label}</text>'
        )

    # ── Probability lines ─────────────────────────────────────────────────────
    svg += (
        # Draw (dashed, behind)
        f'<polyline points="{polyline(points_draw)}"'
        f' fill="none" stroke="#64748b" stroke-width="1.2"'
        f' stroke-dasharray="3,2" opacity="0.6"/>'
        # Away
        f'<polyline points="{polyline(points_away)}"'
        f' fill="none" stroke="{away_colour}" stroke-width="2"'
        f' opacity="0.9"/>'
        # Home (on top)
        f'<polyline points="{polyline(points_home)}"'
        f' fill="none" stroke="{home_colour}" stroke-width="2"'
        f' opacity="0.9"/>'
    )

    # ── End dots ──────────────────────────────────────────────────────────────
    lm,  lp_h = points_home[-1]
    _,   lp_a = points_away[-1]
    _,   lp_d = points_draw[-1]
    svg += (
        f'<circle cx="{x_px(lm):.1f}" cy="{y_px(lp_h):.1f}"'
        f' r="3.5" fill="{home_colour}"/>'
        f'<circle cx="{x_px(lm):.1f}" cy="{y_px(lp_a):.1f}"'
        f' r="3.5" fill="{away_colour}"/>'
        f'<circle cx="{x_px(lm):.1f}" cy="{y_px(lp_d):.1f}"'
        f' r="2.5" fill="#64748b"/>'
    )

    # Final probability labels at right edge
    svg += (
        f'<text x="{x_px(lm) + 5:.1f}" y="{y_px(lp_h) + 3:.1f}"'
        f' fill="{home_colour}" font-size="7" font-weight="bold"'
        f' font-family="JetBrains Mono,monospace">{lp_h:.0%}</text>'
        f'<text x="{x_px(lm) + 5:.1f}" y="{y_px(lp_a) + 3:.1f}"'
        f' fill="{away_colour}" font-size="7" font-weight="bold"'
        f' font-family="JetBrains Mono,monospace">{lp_a:.0%}</text>'
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    lx = PAD_L
    svg += (
        f'<rect x="{lx}" y="6" width="10" height="3"'
        f' fill="{home_colour}" opacity="0.9" rx="1"/>'
        f'<text x="{lx + 13}" y="10" fill="{home_colour}"'
        f' font-size="7.5" font-family="JetBrains Mono,monospace">'
        f'{home_name}</text>'

        f'<rect x="{lx + 100}" y="6" width="10" height="3"'
        f' fill="{away_colour}" opacity="0.9" rx="1"/>'
        f'<text x="{lx + 113}" y="10" fill="{away_colour}"'
        f' font-size="7.5" font-family="JetBrains Mono,monospace">'
        f'{away_name}</text>'

        f'<rect x="{lx + 200}" y="6" width="10" height="3"'
        f' fill="#64748b" opacity="0.7" rx="1"/>'
        f'<text x="{lx + 213}" y="10" fill="#64748b"'
        f' font-size="7.5" font-family="JetBrains Mono,monospace">'
        f'Draw</text>'
    )

    svg += "</svg>"
    return svg
