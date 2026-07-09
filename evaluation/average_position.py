import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from mplsoccer import Pitch

# Ensure repository root is importable when running this script directly
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.coordination_metrics import (
    parse_football_dump,
    pass_success_ratio,
    positional_entropy,
    team_compactness,
)

# --- Constants ---
ROLE_MAP = {
    0: 'GK', 1: 'CB', 2: 'LB', 3: 'RB', 4: 'DM',
    5: 'CM', 6: 'LW', 7: 'RW', 8: 'AM', 9: 'FW'
}

UNITS = {
    "back": [1, 2, 3],
    "midfield": [4, 5, 8],
    "forward": [6, 7, 9]
}

COLORS = {
    "left": "#2196F3",   # Blue
    "right": "#F44336",  # Red
    "background": "#22312b",
    "lines": "#c7d5cc",
    "text": "white"
}

# Football dump eval namespace
FOOTBALL_ACTIONS = [
    'bottom', 'bottom_left', 'bottom_right', 'dribble', 'high_pass', 'idle',
    'left', 'long_pass', 'release_direction', 'release_dribble', 'release_sprint',
    'right', 'short_pass', 'shot', 'sliding', 'sprint', 'top', 'top_left', 'top_right'
]

@dataclass
class TeamData:
    side: str
    positions: np.ndarray  # Shape (N_players, 2)
    roles: List[int]
    color: str

# --- Data Loading & Processing ---

def scale_to_pitch(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scales Google Football coordinates to mplsoccer StatsBomb pitch coordinates.
    GFootball:
        x ∈ [-1, 1]
        y ∈ [-0.42, 0.42]

    StatsBomb:
        x ∈ [0, 120]
        y ∈ [0, 80]
    """
    x = (coords[:, 0] + 1) * 60
    y = (coords[:, 1] + 0.42) * (80 / 0.84)
    return x, y

def process_team(raw_history: np.ndarray, roles: List[int], side: str) -> TeamData:
    """Calculates average positions and returns a TeamData object."""
    avg_pos = np.mean(raw_history, axis=0)

    x, y = scale_to_pitch(avg_pos)
    processed_pos = np.column_stack((x, y))

    return TeamData(
        side=side,
        positions=processed_pos,
        roles=roles,
        color=COLORS[side]
    )

# --- Visualization ---

class PitchVisualizer:
    def __init__(self, title: str):
        self.pitch = Pitch(
            pitch_type='statsbomb',
            pitch_color=COLORS["background"],
            line_color=COLORS["lines"]
        )

        self.fig, self.ax = self.pitch.draw(figsize=(12, 8))
        self.title = title

    def _draw_unit_lines(self, team: TeamData):
        """Draws connecting lines between players in the same tactical unit."""
        for _, role_ids in UNITS.items():
            indices = [i for i, r in enumerate(team.roles) if r in role_ids]

            if len(indices) < 2:
                continue

            sorted_indices = sorted(
                indices,
                key=lambda idx: team.positions[idx, 1]
            )

            coords = team.positions[sorted_indices]

            self.ax.plot(
                coords[:, 0],
                coords[:, 1],
                color=team.color,
                alpha=0.8,
                linestyle='--',
                linewidth=2,
                zorder=1
            )

    def _annotate_roles(self, team: TeamData):
        """Adds role abbreviations (GK, CB, etc.) to player markers."""
        for i, role_id in enumerate(team.roles):
            role_name = ROLE_MAP.get(role_id, f'P{i}')
            pos = team.positions[i]

            self.pitch.annotate(
                role_name,
                xy=(pos[0], pos[1]),
                ax=self.ax,
                color=COLORS["text"],
                fontweight='bold',
                fontsize=10,
                va='center',
                ha='center',
                zorder=3
            )

    def add_team(self, team: TeamData):
        """Plots a team."""
        self.pitch.scatter(
            team.positions[:, 0],
            team.positions[:, 1],
            ax=self.ax,
            color=team.color,
            edgecolors=COLORS["text"],
            s=1000,
            label=f"{team.side.capitalize()} Team",
            zorder=2
        )

        self._draw_unit_lines(team)
        self._annotate_roles(team)

    def _finalize(self):
        plt.title(
            self.title,
            color=COLORS["text"],
            size=16,
            pad=20
        )

        plt.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.05),
            ncol=2
        )

    def save(self, output_path: str, dpi: int = 300):
        """Save figure to disk."""
        self._finalize()

        self.fig.savefig(
            output_path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor=self.fig.get_facecolor()
        )

        print(f"Saved plot: {output_path}")

    def add_metrics_text(self, text: str) -> None:
        self.ax.text(
            0.02,
            0.98,
            text,
            transform=self.ax.transAxes,
            va='top',
            ha='left',
            color='white',
            fontsize=10,
            bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.6')
        )

    def show(self):
        """Display figure."""
        self._finalize()
        plt.show()

# --- Main ---

def main():
    if len(sys.argv) < 2:
        print(
            "Usage:\n"
            "python average_position.py <dump_file> [output_png]"
        )
        sys.exit(1)

    file_path = sys.argv[1]

    # Default output filename
    if len(sys.argv) >= 3:
        output_path = sys.argv[2]
    else:
        output_path = str(Path(file_path).with_suffix(".png"))

    # 1. Load data
    frames = parse_football_dump(file_path)
    left_raw = np.array([frame['observation']['left_team'] for frame in frames])
    right_raw = np.array([frame['observation']['right_team'] for frame in frames])
    left_roles = frames[0]['observation']['left_team_roles']
    right_roles = frames[0]['observation']['right_team_roles']

    # 2. Process
    left_team = process_team(left_raw, left_roles, "left")
    right_team = process_team(right_raw, right_roles, "right")

    left_attempts, left_success, left_psr = pass_success_ratio(frames, 'left')
    right_attempts, right_success, right_psr = pass_success_ratio(frames, 'right')
    _, left_entropy = positional_entropy(frames, 'left')
    _, right_entropy = positional_entropy(frames, 'right')
    left_compactness_mean, left_compactness_std, _, _ = team_compactness(frames, 'left')
    right_compactness_mean, right_compactness_std, _, _ = team_compactness(frames, 'right')

    metrics_text = (
        f'PSR Left: {left_psr:.2f}% ({left_success}/{left_attempts})\n'
        f'PSR Right: {right_psr:.2f}% ({right_success}/{right_attempts})\n'
        f'H_team Left: {left_entropy:.4f} bits\n'
        f'H_team Right: {right_entropy:.4f} bits\n'
        f'Compactness Left: {left_compactness_mean:.4f} ± {left_compactness_std:.4f}\n'
        f'Compactness Right: {right_compactness_mean:.4f} ± {right_compactness_std:.4f}'
    )

    # 3. Visualize
    viz = PitchVisualizer(
        title=f"Average Player Positions\n{Path(file_path).name}"
    )

    viz.add_team(left_team)
    # viz.add_team(right_team)
    viz.add_metrics_text(metrics_text)

    # Save image
    viz.save(output_path)

    # Optional: show window
    viz.show()

if __name__ == "__main__":
    main()