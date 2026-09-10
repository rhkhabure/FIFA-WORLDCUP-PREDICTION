"""
scoreline_matrix.py  —  V4.2
==============================
Renders a scoreline probability heatmap as an inline SVG.

Uses the Dixon-Coles bivariate Poisson joint probability matrix
(same calculation as dc_pregame in main.py) to show the most
likely final scorelines for a given match.

Layout: home goals (0-5) on Y axis, away goals (0-5) on X axis.
Cells coloured by probability -- deeper colour = more likely.
Diagonal = draws, below diagonal = home win, above = away win.

The function build_scoreline_svg() is the only public export.
"""

import json
import numpy as np
from pathlib import Path
from scipy.stats import poisson


def build_scoreline_svg(
    home_name: str,
    away_name: str,
    league: str,
    priors_db: dict,
    home_colour: str = "#14b8a6",
    away_colour: str = "#f43f5e",
    max_goals: int = 5,
    cell_size: int = 52,
    top_n: int = 5,
) -> str:
    """
    Build a scoreline probability heatmap SVG.

    Returns "" if teams aren't found in priors or priors_db is empty.

    Parameters
    ----------
    max_goals   : show scorelines from 0-0 up to max_goals - max_goals
    cell_size   : pixel size of each grid cell
    top_n       : number of top scorelines to highlight in the sidebar
    """
    if not priors_db:
        return ""

    league_data = priors_db.get(league)
    if not league_data:
        return ""

    teams  = league_data["teams"]
    meta   = league_data["meta"]
    gamma  = meta.get("gamma_home_advantage", 1.25)
    rho    = meta.get("rho_draw_correction",  0.0)

    # Alias lookup -- same as dc_pregame
    from v4_backend.feature_builder import TEAM_NAME_ALIASES
    home_key = TEAM_NAME_ALIASES.get(home_name, home_name)
    away_key = TEAM_NAME_ALIASES.get(away_name, away_name)

    all_alpha = [v["alpha"] for v in teams.values()]
    all_beta  = [v["beta"]  for v in teams.values()]
    q25_alpha = float(np.percentile(all_alpha, 25))
    q75_beta  = float(np.percentile(all_beta,  75))

    h = teams.get(home_key, {"alpha": q25_alpha, "beta": q75_beta})
    a = teams.get(away_key, {"alpha": q25_alpha, "beta": q75_beta})

    lam = float(np.clip(h["alpha"] * a["beta"] * gamma, 1e-5, 15.0))
    mu  = float(np.clip(a["alpha"] * h["beta"],          1e-5, 15.0))

    # Build joint matrix
    N  = max_goals + 1
    hp = poisson.pmf(np.arange(N), lam)
    ap = poisson.pmf(np.arange(N), mu)
    joint = np.outer(hp, ap)

    joint[0,0] *= max(1.0 - lam*mu*rho, 1e-5)
    joint[1,0] *= max(1.0 + mu*rho,     1e-5)
    joint[0,1] *= max(1.0 + lam*rho,    1e-5)
    joint[1,1] *= max(1.0 - rho,        1e-5)
    joint /= joint.sum()

    # Scale remaining probability so displayed cells sum cleanly
    max_prob = float(joint.max())

    # ── Layout constants ──────────────────────────────────────────────────────
    PAD_L   = 28   # left padding for home-goals axis labels
    PAD_T   = 28   # top padding for away-goals axis labels

    grid_w  = N * cell_size
    grid_h  = N * cell_size
    # Top scorelines stacked below the grid (no sidebar)
    LIST_H  = top_n * 18 + 36   # height for scoreline list below grid
    total_w = PAD_L + grid_w + 8
    total_h = PAD_T + grid_h + 20 + LIST_H

    svg = (
        f'<svg width="100%" viewBox="0 0 {total_w} {total_h}"'
        f' xmlns="http://www.w3.org/2000/svg"'
        f' style="display:block;overflow:visible">'
        f'<rect width="{total_w}" height="{total_h}" fill="#0b1220" rx="6"/>'
    )

    # ── Axis labels ───────────────────────────────────────────────────────────
    # Home team label (left, vertical)
    home_short = " ".join(home_name.split()[:2])
    away_short = " ".join(away_name.split()[:2])

    svg += (
        f'<text x="{PAD_L + grid_w // 2}" y="{total_h - 4}"'
        f' fill="{away_colour}" font-size="8" text-anchor="middle"'
        f' font-family="JetBrains Mono,monospace"'
        f' font-weight="bold">{away_short} goals →</text>'
        f'<text x="8" y="{PAD_T + grid_h // 2}"'
        f' fill="{home_colour}" font-size="8" text-anchor="middle"'
        f' font-family="JetBrains Mono,monospace" font-weight="bold"'
        f' transform="rotate(-90,8,{PAD_T + grid_h // 2})"'
        f'>{home_short} goals →</text>'
    )

    # Away goals numbers (top)
    for j in range(N):
        cx = PAD_L + j * cell_size + cell_size // 2
        svg += (
            f'<text x="{cx}" y="{PAD_T - 6}"'
            f' fill="#64748b" font-size="9" text-anchor="middle"'
            f' font-family="JetBrains Mono,monospace">{j}</text>'
        )

    # Home goals numbers (left)
    for i in range(N):
        cy = PAD_T + i * cell_size + cell_size // 2 + 3
        svg += (
            f'<text x="{PAD_L - 6}" y="{cy}"'
            f' fill="#64748b" font-size="9" text-anchor="end"'
            f' font-family="JetBrains Mono,monospace">{i}</text>'
        )

    # ── Grid cells ────────────────────────────────────────────────────────────
    for i in range(N):          # i = home goals (rows)
        for j in range(N):     # j = away goals (cols)
            prob = float(joint[i, j])
            intensity = prob / max_prob   # 0..1 relative brightness

            # Colour by outcome region
            if i > j:    # home win zone
                r = int(0x14 + (0xC8 - 0x14) * intensity)
                g = int(0x20 + (0x10 - 0x20) * intensity)
                b = int(0x20 + (0x2E - 0x20) * intensity)
                cell_fill = f"#{r:02x}{g:02x}{b:02x}"
            elif j > i:  # away win zone
                r = int(0x14 + (0x10 - 0x14) * intensity)
                g = int(0x20 + (0x20 - 0x20) * intensity)
                b = int(0x20 + (0xB2 - 0x20) * intensity)
                cell_fill = f"#{r:02x}{g:02x}{b:02x}"
            else:        # draw diagonal
                v = int(0x20 + (0x94 - 0x20) * intensity)
                cell_fill = f"#{v:02x}{v:02x}{v:02x}"

            x = PAD_L + j * cell_size
            y = PAD_T + i * cell_size

            # Highlight the most likely cell in each zone
            is_top = prob >= max_prob * 0.85
            stroke = "white" if is_top else "#1e293b"
            sw     = "1.5"   if is_top else "0.5"

            svg += (
                f'<rect x="{x}" y="{y}"'
                f' width="{cell_size}" height="{cell_size}"'
                f' fill="{cell_fill}" stroke="{stroke}"'
                f' stroke-width="{sw}"/>'
            )

            # Probability text inside cell
            pct_str = f"{prob:.1%}"
            text_col = "white" if intensity > 0.4 else "#64748b"
            svg += (
                f'<text x="{x + cell_size//2}" y="{y + cell_size//2 - 5}"'
                f' fill="{text_col}" font-size="9" text-anchor="middle"'
                f' font-family="JetBrains Mono,monospace"'
                f' font-weight="{"bold" if is_top else "normal"}">'
                f'{i}-{j}</text>'
                f'<text x="{x + cell_size//2}" y="{y + cell_size//2 + 8}"'
                f' fill="{text_col}" font-size="8" text-anchor="middle"'
                f' font-family="JetBrains Mono,monospace"'
                f' opacity="0.85">{pct_str}</text>'
            )

    # ── Top scorelines below grid ─────────────────────────────────────────────
    all_scores = sorted(
        [(float(joint[i,j]), i, j) for i in range(N) for j in range(N)],
        reverse=True
    )

    p_home = float(np.tril(joint, -1).sum())
    p_draw = float(np.trace(joint))
    p_away = float(np.triu(joint, +1).sum())

    list_y = PAD_T + grid_h + 28   # start below the axis label

    # Outcome totals header
    svg += (
        f'<line x1="{PAD_L}" y1="{list_y - 8}" x2="{total_w - 4}" y2="{list_y - 8}"'
        f' stroke="#1e293b" stroke-width="0.8"/>'
        f'<text x="{PAD_L}" y="{list_y + 2}"'
        f' fill="{home_colour}" font-size="8" font-weight="bold"'
        f' font-family="JetBrains Mono,monospace">H {p_home:.0%}</text>'
        f'<text x="{PAD_L + 52}" y="{list_y + 2}"'
        f' fill="#64748b" font-size="8"'
        f' font-family="JetBrains Mono,monospace">D {p_draw:.0%}</text>'
        f'<text x="{PAD_L + 104}" y="{list_y + 2}"'
        f' fill="{away_colour}" font-size="8" font-weight="bold"'
        f' font-family="JetBrains Mono,monospace">A {p_away:.0%}</text>'
        f'<text x="{total_w - 8}" y="{list_y + 2}"'
        f' fill="#475569" font-size="7.5" text-anchor="end"'
        f' font-family="JetBrains Mono,monospace">xG {lam:.2f}–{mu:.2f}</text>'
    )

    # Top scorelines in a horizontal row
    bar_total_w = total_w - PAD_L - 8
    for rank, (prob, i, j) in enumerate(all_scores[:top_n]):
        ry = list_y + 14 + rank * 18
        outcome = "H" if i > j else ("A" if j > i else "D")
        colour  = (home_colour if outcome == "H"
                   else away_colour if outcome == "A"
                   else "#64748b")
        bar_w = int((bar_total_w - 60) * prob / all_scores[0][0])
        svg += (
            f'<text x="{PAD_L}" y="{ry}"'
            f' fill="#334155" font-size="8"'
            f' font-family="JetBrains Mono,monospace">#{rank+1}</text>'
            f'<text x="{PAD_L + 18}" y="{ry}"'
            f' fill="{colour}" font-size="9" font-weight="bold"'
            f' font-family="JetBrains Mono,monospace">{i}-{j}</text>'
            f'<rect x="{PAD_L + 38}" y="{ry - 9}"'
            f' width="{bar_w}" height="9"'
            f' fill="{colour}" opacity="0.5" rx="2"/>'
            f'<text x="{PAD_L + 42 + bar_w}" y="{ry}"'
            f' fill="{colour}" font-size="7.5"'
            f' font-family="JetBrains Mono,monospace">{prob:.1%}</text>'
        )

    svg += "</svg>"
    return svg
