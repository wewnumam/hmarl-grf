import argparse
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


FOOTBALL_ACTIONS = [
    'bottom', 'bottom_left', 'bottom_right', 'dribble', 'high_pass', 'idle',
    'left', 'long_pass', 'release_direction', 'release_dribble', 'release_sprint',
    'right', 'short_pass', 'shot', 'sliding', 'sprint', 'top', 'top_left', 'top_right'
]
PASS_ACTIONS = {'short_pass', 'long_pass', 'high_pass'}


# --- Data loading ---

def get_eval_namespace() -> Dict[str, Any]:
    namespace = {action: action for action in FOOTBALL_ACTIONS}
    namespace.update({'array': np.array, 'uint8': np.uint8})
    return namespace


def parse_football_dump(file_path: str) -> List[Dict[str, Any]]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = eval(f.read(), get_eval_namespace())
    except Exception as e:
        raise RuntimeError(f'Failed to parse dump file: {e}')

    if not isinstance(data, list):
        raise ValueError('Expected dump file to contain a list of frames.')

    return data


# --- Coordinate transformations ---

def scale_to_pitch(coords: np.ndarray) -> np.ndarray:
    x = (coords[:, 0] + 1.0) * 60.0
    y = (coords[:, 1] + 0.42) * (80.0 / 0.84)
    return np.column_stack((x, y))


# --- Metrics ---

def pass_success_ratio(frames: List[Dict[str, Any]], team_side: str = 'left', success_window: int = 5) -> Tuple[int, int, float]:
    assert team_side in ('left', 'right')
    total_attempts = 0
    successful_passes = 0
    team_id = 0 if team_side == 'left' else 1

    for idx, frame in enumerate(frames[:-1]):
        obs = frame['observation']
        owned_team = obs.get('ball_owned_team', -1)
        owned_player = obs.get('ball_owned_player', -1)

        if owned_team != team_id:
            continue
        if owned_player is None or owned_player < 0:
            continue

        actions = frame['debug'].get('action', [])
        if not isinstance(actions, (list, tuple)):
            continue

        if owned_player >= len(actions):
            continue

        if actions[owned_player] not in PASS_ACTIONS:
            continue

        total_attempts += 1
        success = False

        for future_idx in range(idx + 1, min(idx + 1 + success_window, len(frames))):
            future_obs = frames[future_idx]['observation']
            future_team = future_obs.get('ball_owned_team', -1)
            future_player = future_obs.get('ball_owned_player', -1)

            if future_team == team_id and future_player is not None and future_player >= 0 and future_player != owned_player:
                success = True
                break

            if future_team == 1 - team_id:
                break

        if success:
            successful_passes += 1

    ratio = (successful_passes / total_attempts * 100.0) if total_attempts > 0 else 0.0
    return total_attempts, successful_passes, ratio


def positional_entropy(frames: List[Dict[str, Any]], team_side: str = 'left', grid_shape: Tuple[int, int] = (6, 4)) -> Tuple[np.ndarray, float]:
    assert team_side in ('left', 'right')
    nx, ny = grid_shape
    num_zones = nx * ny

    positions = np.array([frame['observation'][f'{team_side}_team'] for frame in frames])
    positions_scaled = np.stack([scale_to_pitch(positions[t]) for t in range(positions.shape[0])])

    counts = np.zeros((positions.shape[1], num_zones), dtype=float)

    x_edges = np.linspace(0.0, 120.0, nx + 1)
    y_edges = np.linspace(0.0, 80.0, ny + 1)

    for t in range(positions_scaled.shape[0]):
        for player_idx in range(positions_scaled.shape[1]):
            x, y = positions_scaled[t, player_idx]
            xi = min(nx - 1, np.searchsorted(x_edges, x, side='right') - 1)
            yi = min(ny - 1, np.searchsorted(y_edges, y, side='right') - 1)
            if xi < 0:
                xi = 0
            if yi < 0:
                yi = 0
            zone_idx = yi * nx + xi
            counts[player_idx, zone_idx] += 1

    entropies = np.zeros(positions.shape[1], dtype=float)
    for player_idx in range(positions.shape[1]):
        total = counts[player_idx].sum()
        if total <= 0:
            entropies[player_idx] = 0.0
            continue

        probs = counts[player_idx] / total
        mask = probs > 0
        entropies[player_idx] = -np.sum(probs[mask] * np.log2(probs[mask]))

    team_entropy = float(np.mean(entropies))
    return entropies, team_entropy


def team_compactness(frames: List[Dict[str, Any]], team_side: str = 'left') -> Tuple[float, float, float]:
    assert team_side in ('left', 'right')
    positions = np.array([frame['observation'][f'{team_side}_team'] for frame in frames])
    positions_scaled = np.stack([scale_to_pitch(positions[t]) for t in range(positions.shape[0])])

    densities = []
    for t in range(positions_scaled.shape[0]):
        frame_positions = positions_scaled[t]
        centroid = frame_positions.mean(axis=0)
        deviations = frame_positions - centroid
        squared_distances = np.sum(deviations ** 2, axis=1)
        rho = math.sqrt(np.mean(squared_distances))
        densities.append(rho)

    densities = np.array(densities)
    return float(densities.mean()), float(densities.std()), float(densities.min()), float(densities.max())


def format_entropy(entropy_value: float) -> str:
    return f'{entropy_value:.4f} bits'


def main() -> None:
    parser = argparse.ArgumentParser(description='Compute coordination metrics from Google Research Football dump data.')
    parser.add_argument('dump_file', type=str, help='Path to episode dump file')
    parser.add_argument('--team', choices=['left', 'right', 'both'], default='left', help='Team to evaluate')
    parser.add_argument('--grid', type=str, default='6x4', help='Grid shape for positional entropy as NxM')
    args = parser.parse_args()

    dump_path = Path(args.dump_file)
    if not dump_path.exists():
        print(f'File not found: {dump_path}')
        sys.exit(1)

    try:
        frames = parse_football_dump(str(dump_path))
    except Exception as exc:
        print(exc)
        sys.exit(1)

    grid_parts = args.grid.split('x')
    if len(grid_parts) != 2:
        print('Grid must be specified as NxM, for example 6x4')
        sys.exit(1)

    grid_shape = (int(grid_parts[0]), int(grid_parts[1]))
    teams = ['left'] if args.team != 'both' else ['left', 'right']

    print(f'Coordination metrics for: {dump_path.name}')
    print(f'Positional entropy grid: {grid_shape[0]} x {grid_shape[1]} zones')
    print(f'Total frames: {len(frames)}')
    print('')

    for team_side in teams:
        total_attempts, successful_passes, ratio = pass_success_ratio(frames, team_side)
        entropies, team_entropy = positional_entropy(frames, team_side, grid_shape)
        compactness_mean, compactness_std, compactness_min, compactness_max = team_compactness(frames, team_side)

        print(f'Team: {team_side.capitalize()}')
        print(f'  Pass attempts: {total_attempts}')
        print(f'  Successful passes: {successful_passes}')
        print(f'  Pass Success Ratio (PSR): {ratio:.2f}%')
        print(f'  Team positional entropy (H_team): {format_entropy(team_entropy)}')
        print(f'  Team compactness (mean rho_tc): {compactness_mean:.4f}')
        print(f'  Compactness std: {compactness_std:.4f}')
        print(f'  Compactness min/max: {compactness_min:.4f} / {compactness_max:.4f}')
        print('')

    if args.team == 'left':
        print('Interpretasi:')
        print('  - PSR menilai seberapa sering operan berhasil ke rekan satu tim tanpa kehilangan kepemilikan.')
        print('  - Entropi posisi rendah menunjukkan disiplin posisi, sedangkan nilai tinggi menandakan pergerakan posisi yang lebih acak.')
        print('  - Compactness kecil berarti tim rapat; besar berarti tim lebih menyebar.')


if __name__ == '__main__':
    main()
