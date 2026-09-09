"""
timeline.py  —  V4.2
======================
Win probability timeline for the Live Match page.

Builds the minute-by-minute probability curve using football_v4.pth,
then renders it as a self-contained inline SVG chart.

No JavaScript charting library needed -- pure SVG paths.
Goal events are marked as vertical dashed lines with scorer labels.

The function build_match_timeline_svg() is the only public export.
"""

import sys
from pathlib import Path

import numpy as np
import torch

# ── Internal helpers ──────────────────────────────────────────────────────────

def _simulate_score(goals: list[dict], home_name: str, minute: int) -> tuple[int, int]:
    """
    Reconstruct the scoreline at a given minute from the goals list.
    goals: list of dicts with 'minute' (int) and 'is_home' (bool).
    """
    hs = sum(1 for g in goals if g["minute"] <= minute and g["is_home"])
    as_ = sum(1 for g in goals if g["minute"] <= minute and not g["is_home"])
    return hs, as_


def _parse_goals_for_timeline(raw_goals: list[dict],
                               home_name: str) -> list[dict]:
    """
    Convert the raw goals list from footballdata.py into a clean list
    of {'minute': int, 'is_home': bool, 'scorer': str} dicts.
    """
    parsed = []
    for g in raw_goals:
        # raw_goals come from _parse_goals() which stores:
        # {'time': "56'", 'detail': 'Casemiro (West Ham) - REGULAR'}
        time_str = g.get("time", "0'").replace("'", "").split("+")[0]
        try:
            minute = int(time_str.strip())
        except ValueError:
            continue

        detail = g.get("detail", "")
        # 'detail' format: "Scorer (Team) - TYPE"
        is_home = f"({home_name})" in detail or \
                  home_name.lower().split()[0] in detail.lower()
        scorer = detail.split("(")[0].strip() if "(" in detail else "Goal"

        parsed.append({
            "minute" : minute,
            "is_home": is_home,
            "scorer" : scorer,
        })
    return sorted(parsed, key=lambda x: x["minute"])


def build_match_timeline_svg(
    home_name: str,
    away_name: str,
    league: str,
    raw_goals: list[dict],
    home_colour: str,
    away_colour: str,
    dc_lookup,
    nn_model,
    nn_scaler,
    nn_T: float,
    width: int = 700,
    height: int = 180,
) -> str:
    """
    Build a win probability timeline SVG for a match.

    Returns an SVG string ready to embed in match.html via {{ timeline_svg | safe }}.
    Returns "" if the match has no goals yet (pre-game) or model is unavailable.
    """
    if nn_model is None or nn_scaler is None or dc_lookup is None:
        return ""

    goals = _parse_goals_for_timeline(raw_goals, home_name)

    # Checkpoints: every 5 minutes + each goal minute + 0 and 90
    minutes = sorted(set([0] + list(range(5, 91, 5)) +
                         [g["minute"] for g in goals] + [90]))

    points_home, points_draw, points_away = [], [], []

    for m in minutes:
        hs, as_ = _simulate_score(goals, home_name, m)
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

        points_home.append((m, float(p[2])))   # p_home
        points_draw.append((m, float(p[1])))   # p_draw
        points_away.append((m, float(p[0])))   # p_away

    if not points_home:
        return ""

    # ── SVG layout ────────────────────────────────────────────────────────────
    PAD_L, PAD_R, PAD_T, PAD_B = 32, 16, 20, 32
    W = width  - PAD_L - PAD_R
    H = height - PAD_T - PAD_B

    def x_px(minute):
        return PAD_L + (minute / 90) * W

    def y_px(prob):
        return PAD_T + (1 - prob) * H   # prob=1 at top, 0 at bottom

    def pts_to_polyline(pts):
        return " ".join(f"{x_px(m):.1f},{y_px(p):.1f}" for m, p in pts)

    svg = (
        f'<svg width="100%" viewBox="0 0 {width} {height}"'
        f' xmlns="http://www.w3.org/2000/svg"'
        f' style="display:block;overflow:visible">'

        # Background
        f'<rect width="{width}" height="{height}"'
        f' fill="#0b1220" rx="6"/>'

        # Horizontal grid lines at 25%, 50%, 75%
    )
    for pct in [0.25, 0.50, 0.75]:
        yg = y_px(pct)
        svg += (
            f'<line x1="{PAD_L}" y1="{yg:.1f}"'
            f' x2="{width-PAD_R}" y2="{yg:.1f}"'
            f' stroke="#1e293b" stroke-width="0.8" stroke-dasharray="3,3"/>'
            f'<text x="{PAD_L - 4}" y="{yg + 3:.1f}"'
            f' fill="#475569" font-size="7" text-anchor="end"'
            f' font-family="JetBrains Mono,monospace">{int(pct*100)}%</text>'
        )

    # Minute axis labels
    for m in [0, 15, 30, 45, 60, 75, 90]:
        xg = x_px(m)
        svg += (
            f'<line x1="{xg:.1f}" y1="{PAD_T}"'
            f' x2="{xg:.1f}" y2="{PAD_T + H}"'
            f' stroke="#1e293b" stroke-width="0.5"/>'
            f'<text x="{xg:.1f}" y="{PAD_T + H + 10}"'
            f' fill="#475569" font-size="7" text-anchor="middle"'
            f' font-family="JetBrains Mono,monospace">{m}\'</text>'
        )

    # Goal event vertical lines
    for g in goals:
        xg = x_px(g["minute"])
        colour = home_colour if g["is_home"] else away_colour
        scorer_short = g["scorer"].split()[-1] if g["scorer"] else "Goal"
        svg += (
            f'<line x1="{xg:.1f}" y1="{PAD_T}"'
            f' x2="{xg:.1f}" y2="{PAD_T + H}"'
            f' stroke="{colour}" stroke-width="1"'
            f' stroke-dasharray="4,3" opacity="0.7"/>'
            f'<text x="{xg + 2:.1f}" y="{PAD_T + 8}"'
            f' fill="{colour}" font-size="6"'
            f' font-family="JetBrains Mono,monospace"'
            f' opacity="0.9">{scorer_short}</text>'
        )

    # Draw probability lines (draw first so it's behind)
    svg += (
        f'<polyline points="{pts_to_polyline(points_draw)}"'
        f' fill="none" stroke="#64748b" stroke-width="1.2"'
        f' stroke-dasharray="3,2" opacity="0.7"/>'

        f'<polyline points="{pts_to_polyline(points_away)}"'
        f' fill="none" stroke="{away_colour}" stroke-width="1.8"'
        f' opacity="0.85"/>'

        f'<polyline points="{pts_to_polyline(points_home)}"'
        f' fill="none" stroke="{home_colour}" stroke-width="1.8"'
        f' opacity="0.85"/>'
    )

    # End-of-match dots
    if points_home:
        lm, lp_h = points_home[-1]
        _, lp_a   = points_away[-1]
        _, lp_d   = points_draw[-1]
        svg += (
            f'<circle cx="{x_px(lm):.1f}" cy="{y_px(lp_h):.1f}"'
            f' r="3" fill="{home_colour}"/>'
            f'<circle cx="{x_px(lm):.1f}" cy="{y_px(lp_a):.1f}"'
            f' r="3" fill="{away_colour}"/>'
            f'<circle cx="{x_px(lm):.1f}" cy="{y_px(lp_d):.1f}"'
            f' r="2.5" fill="#64748b"/>'
        )

    # Legend
    legend_x = PAD_L
    svg += (
        f'<rect x="{legend_x}" y="4" width="8" height="3"'
        f' fill="{home_colour}" opacity="0.9" rx="1"/>'
        f'<text x="{legend_x + 10}" y="8" fill="{home_colour}"'
        f' font-size="7" font-family="JetBrains Mono,monospace">{home_name}</text>'

        f'<rect x="{legend_x + 80}" y="4" width="8" height="3"'
        f' fill="{away_colour}" opacity="0.9" rx="1"/>'
        f'<text x="{legend_x + 92}" y="8" fill="{away_colour}"'
        f' font-size="7" font-family="JetBrains Mono,monospace">{away_name}</text>'

        f'<rect x="{legend_x + 160}" y="4" width="8" height="3"'
        f' fill="#64748b" opacity="0.7" rx="1"/>'
        f'<text x="{legend_x + 172}" y="8" fill="#64748b"'
        f' font-size="7" font-family="JetBrains Mono,monospace">Draw</text>'
    )

    svg += "</svg>"
    return svg
