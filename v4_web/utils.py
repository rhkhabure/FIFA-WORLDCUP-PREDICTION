import numpy as np

def generate_pitch_svg(home_formation="4-3-3", away_formation="4-2-3-1", home_players=None, away_players=None):
    """Generates a native horizontal full-pitch. Quant Theme."""
    if home_players is None:
        home_players = [f"H{i}" for i in range(11)]
    if away_players is None:
        away_players = [f"A{i}" for i in range(11)]
        
    def parse_formation(fmt_str):
        if not fmt_str or fmt_str == "0-0": return []
        return [1] + [int(x) for x in fmt_str.split("-")]
        
    h_lines = parse_formation(home_formation)
    a_lines = parse_formation(away_formation)
    
    W, H = 160, 100
    
    # Use the specific colors requested: teal for home, coral for away, dark slate for pitch
    svg = f"""<svg width="100%" height="100%" viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">
        <defs>
            <circle id="jersey-home" cx="0" cy="0" r="3" fill="#06b6d4" stroke="#0b0f19" stroke-width="0.8"/>
            <circle id="jersey-away" cx="0" cy="0" r="3" fill="#f43f5e" stroke="#0b0f19" stroke-width="0.8"/>
        </defs>
        <rect width="{W}" height="{H}" fill="#111827" rx="4" ry="4" stroke="#1e293b" stroke-width="1.5"/>
        <rect x="5" y="5" width="150" height="90" fill="none" stroke="#334155" stroke-width="0.5"/>
        
        <!-- Field markings -->
        <line x1="80" y1="5" x2="80" y2="95" stroke="#334155" stroke-width="0.5"/>
        <circle cx="80" cy="50" r="15" fill="none" stroke="#334155" stroke-width="0.5"/>
        <circle cx="80" cy="50" r="1" fill="#334155" />
        
        <rect x="5" y="22" width="22" height="56" fill="none" stroke="#334155" stroke-width="0.5"/>
        <rect x="133" y="22" width="22" height="56" fill="none" stroke="#334155" stroke-width="0.5"/>
        <rect x="5" y="37" width="8" height="26" fill="none" stroke="#334155" stroke-width="0.5"/>
        <rect x="147" y="37" width="8" height="26" fill="none" stroke="#334155" stroke-width="0.5"/>
        <line x1="5" y1="44" x2="5" y2="56" stroke="#64748b" stroke-width="1.5" />
        <line x1="155" y1="44" x2="155" y2="56" stroke="#64748b" stroke-width="1.5" />
    """
    
    # Draw Home Players (Left side)
    if h_lines:
        x_steps = len(h_lines)
        p_idx = 0
        for col_idx, num_players in enumerate(h_lines):
            x = 15 + (60 / max(1, x_steps - 1)) * col_idx
            for row_idx in range(num_players):
                y = 10 + (80 / (num_players + 1)) * (row_idx + 1)
                name = home_players[p_idx] if p_idx < len(home_players) else ""
                svg += f"""
                    <use href="#jersey-home" x="{x}" y="{y}" />
                    <text x="{x}" y="{y+5.5}" fill="#94a3b8" font-family="'JetBrains Mono', monospace" font-size="2.5px" text-anchor="middle">{name}</text>
                """
                p_idx += 1

    # Draw Away Players (Right side)
    if a_lines:
        x_steps = len(a_lines)
        p_idx = 0
        for col_idx, num_players in enumerate(a_lines):
            x = 145 - (60 / max(1, x_steps - 1)) * col_idx
            for row_idx in range(num_players):
                y = 10 + (80 / (num_players + 1)) * (row_idx + 1)
                name = away_players[p_idx] if p_idx < len(away_players) else ""
                svg += f"""
                    <use href="#jersey-away" x="{x}" y="{y}" />
                    <text x="{x}" y="{y+5.5}" fill="#94a3b8" font-family="'JetBrains Mono', monospace" font-size="2.5px" text-anchor="middle">{name}</text>
                """
                p_idx += 1
            
    svg += "</svg>"
    return svg
