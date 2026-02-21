import matplotlib.pyplot as plt
import numpy as np
import sys
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from mplsoccer import Pitch

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

def get_eval_namespace() -> Dict[str, Any]:
    """Returns a restricted namespace for safe eval of football dump files."""
    namespace = {action: action for action in FOOTBALL_ACTIONS}
    namespace.update({"array": np.array, "uint8": np.uint8})
    return namespace

def parse_football_dump(file_path: str) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    """Loads and parses the football episode dump file."""
    try:
        with open(file_path, 'r') as f:
            data = eval(f.read(), get_eval_namespace())
    except Exception as e:
        print(f"Error parsing file: {e}")
        sys.exit(1)

    left_raw = np.array([frame['observation']['left_team'] for frame in data])
    right_raw = np.array([frame['observation']['right_team'] for frame in data])
    left_roles = data[0]['observation']['left_team_roles']
    right_roles = data[0]['observation']['right_team_roles']
    
    return left_raw, right_raw, left_roles, right_roles

def scale_to_pitch(coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Scales Google Football coordinates (-1 to 1) to mplsoccer Pitch (0-120, 0-80)."""
    # X: (-1, 1) -> (0, 120), Y: (-0.42, 0.42) -> (0, 80)
    x = (coords[:, 0] + 1) * 60
    y = (coords[:, 1] + 0.42) * (80 / 0.84)
    return x, y

def process_team(raw_history: np.ndarray, roles: List[int], side: str) -> TeamData:
    """Calculates average positions and returns a TeamData object."""
    avg_pos = np.mean(raw_history, axis=0)
    x, y = scale_to_pitch(avg_pos)
    # Re-combine into (N, 2)
    processed_pos = np.column_stack((x, y))
    return TeamData(side=side, positions=processed_pos, roles=roles, color=COLORS[side])

# --- Visualization ---

class PitchVisualizer:
    def __init__(self, title: str):
        self.pitch = Pitch(pitch_type='statsbomb', pitch_color=COLORS["background"], line_color=COLORS["lines"])
        self.fig, self.ax = self.pitch.draw(figsize=(12, 8))
        self.title = title

    def _draw_unit_lines(self, team: TeamData):
        """Draws connecting lines between players in the same tactical unit."""
        for unit_name, role_ids in UNITS.items():
            indices = [i for i, r in enumerate(team.roles) if r in role_ids]
            if len(indices) < 2:
                continue

            # Sort indices by Y coordinate for a clean line across the pitch
            sorted_indices = sorted(indices, key=lambda idx: team.positions[idx, 1])
            coords = team.positions[sorted_indices]
            
            self.ax.plot(coords[:, 0], coords[:, 1], color=team.color, 
                        alpha=0.3, linestyle='--', linewidth=2, zorder=1)

    def _annotate_roles(self, team: TeamData):
        """Adds role abbreviations (GK, CB, etc.) to player markers."""
        for i, role_id in enumerate(team.roles):
            role_name = ROLE_MAP.get(role_id, f'P{i}')
            pos = team.positions[i]
            self.pitch.annotate(role_name, xy=(pos[0], pos[1]), ax=self.ax, 
                               color=COLORS["text"], fontweight='bold', va='center', ha='center',
                               fontsize=10, zorder=3)

    def add_team(self, team: TeamData):
        """Main entry point to plot a team on the pitch."""
        # Plot markers
        self.pitch.scatter(team.positions[:, 0], team.positions[:, 1], ax=self.ax, 
                          color=team.color, edgecolors=COLORS["text"], s=350, 
                          label=f"{team.side.capitalize()} Team", zorder=2)
        
        # Add tactical lines and text
        self._draw_unit_lines(team)
        self._annotate_roles(team)

    def show(self):
        plt.title(self.title, color=COLORS["text"], size=16, pad=20)
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
        plt.show()

# --- Main ---

def main():
    if len(sys.argv) < 2:
        print("Usage: python average_position.py <path_to_dump_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    
    # 1. Load Data
    left_raw, right_raw, left_roles, right_roles = parse_football_dump(file_path)
    
    # 2. Process Data
    left_team = process_team(left_raw, left_roles, "left")
    right_team = process_team(right_raw, right_roles, "right")
    
    # 3. Visualize
    viz = PitchVisualizer(title=f'Average Player Positions\n{file_path}')
    viz.add_team(left_team)
    viz.add_team(right_team)
    viz.show()

if __name__ == "__main__":
    main()