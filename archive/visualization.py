
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mplsoccer import Pitch
import numpy as np
from mesa_model import FootballModel

def run_simulation_and_visualize(steps=100, output_file='grf_mesa_simulation.mp4'):
    # Initialize Model
    model = FootballModel()
    
    # Storage for history to animate later
    history = []
    
    print(f"Running simulation for {steps} steps...")
    for i in range(steps):
        done = model.step()
        
        # Snapshot state
        # We need to copy positions
        # Sort agents by player_idx to ensure consistent role mapping
        left_agents = sorted([agent for agent in model.agents if agent.team_id == 0], key=lambda x: x.player_idx)
        right_agents = sorted([agent for agent in model.agents if agent.team_id == 1], key=lambda x: x.player_idx)
        
        left_pos = [agent.pos for agent in left_agents]
        right_pos = [agent.pos for agent in right_agents]
        ball_pos = model.ball_pos
        
        history.append({
            'left': np.array(left_pos),
            'right': np.array(right_pos),
            'ball': np.array(ball_pos)
        })
        
        if done:
            print("Episode done early.")
            break
            
    print("Simulation complete. Generating visualization...")
    
    # === Visualization ===
    
    # GRF Field Dimensions: 105x68 meters usually, but coordinate system is norm [-1, 1].
    # mplsoccer default is 105x68 (statsbomb) or 120x80.
    # We need to map GRF [-1, 1] (x) and [-0.42, 0.42] (y) to mplsoccer pitch.
    # Default mplsoccer pitch (statsbomb) is 0-120 (x), 0-80 (y).
    
    pitch = Pitch(pitch_type='statsbomb', pitch_color='#aabb97', line_color='white',
                  stripe=True)
    
    fig, ax = pitch.draw(figsize=(10, 6))
    
    # Initial Plot Objects
    left_scat = pitch.scatter([], [], ax=ax, c='red', s=100, label='Left Team', edgecolors='black', zorder=10)
    right_scat = pitch.scatter([], [], ax=ax, c='blue', s=100, label='Right Team', edgecolors='black', zorder=10)
    ball_scat = pitch.scatter([], [], ax=ax, c='white', s=80, marker='o', edgecolors='black', label='Ball', zorder=10)
    
    # Tactical Lines (Back 4, Mid 3, Fwd 3)
    # Styles for lines
    line_style = {'color': 'red', 'linestyle': '--', 'alpha': 0.6, 'linewidth': 2}
    
    line_def, = ax.plot([], [], **line_style, label='Defense')
    line_mid, = ax.plot([], [], **line_style, label='Midfield')
    line_fwd, = ax.plot([], [], **line_style, label='Forward')

    # Coordinate Transformation Function
    def transform_coords(grf_coords, reverse_y=True):
        """
        Transform GRF [-1, 1] x [-0.42, 0.42] to Statsbomb [0, 120] x [0, 80]
        """
        if len(grf_coords) == 0: return np.array([]), np.array([])
        
        # GRF X: -1 to 1 -> 0 to 120
        # x_new = (x_old + 1) / 2 * 120
        x = (grf_coords[:, 0] + 1) / 2 * 120
        
        # GRF Y: -0.42 to 0.42 -> 0 to 80 (roughly)
        y = (grf_coords[:, 1] + 0.42) / 0.84 * 80
        
        if reverse_y:
            y = 80 - y
            
        return x, y

    def get_sorted_line_coords(x, y, indices):
        """Helper to get sorted coordinates for a subset of players."""
        # Filter by indices
        sub_x = x[indices]
        sub_y = y[indices]
        # Sort by Y to connect lines cleanly (top to bottom)
        sort_idx = np.argsort(sub_y)
        return sub_x[sort_idx], sub_y[sort_idx]

    def update(frame_idx):
        data = history[frame_idx]
        
        # Update Left Team
        lx, ly = transform_coords(data['left'])
        left_scat.set_offsets(np.c_[lx, ly])
        
        # Update Right Team
        rx, ry = transform_coords(data['right'])
        right_scat.set_offsets(np.c_[rx, ry])
        
        # Update Ball
        bx, by = transform_coords(data['ball'].reshape(1, 2))
        ball_scat.set_offsets(np.c_[bx, by])
        
        # Update Lines
        # Indices: GK=0, Def=1-4, Mid=5-7, Fwd=8-10
        # Check if we have enough players
        if len(lx) >= 11:
            # Defense
            dx, dy = get_sorted_line_coords(lx, ly, [1, 2, 3, 4])
            line_def.set_data(dx, dy)
            
            # Midfield
            mx, my = get_sorted_line_coords(lx, ly, [5, 6, 7])
            line_mid.set_data(mx, my)
            
            # Forward
            fx, fy = get_sorted_line_coords(lx, ly, [8, 9, 10])
            line_fwd.set_data(fx, fy)
        
        ax.set_title(f"Step {frame_idx}")
        return left_scat, right_scat, ball_scat, line_def, line_mid, line_fwd

    anim = animation.FuncAnimation(fig, update, frames=len(history), interval=100, blit=True)
    
    # Save
    try:
        anim.save(output_file, writer='ffmpeg', fps=10)
        print(f"Animation saved to {output_file}")
    except Exception as e:
        print(f"Failed to save video: {e}")
        # Fallback to gif if ffmpeg missing
        anim.save(output_file.replace('.mp4', '.gif'), writer='pillow', fps=10)
        print(f"Animation saved to {output_file.replace('.mp4', '.gif')}")

if __name__ == "__main__":
    run_simulation_and_visualize()
