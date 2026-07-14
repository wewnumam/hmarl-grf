# HMARL-GRF: Hybrid Multi-Agent Reinforcement Learning for Google Research Football

This repository contains tools and scripts for training multi-agent reinforcement learning models in the [Google Research Football (GRF)](https://github.com/google-research/football) environment. It focuses on 11v11 scenarios using algorithms like PPO and A2C via Stable Baselines3.

## 🚀 Quick Start

### Prerequisites
- **Docker Desktop** (with WSL2 backend recommended)
- **NVIDIA Container Toolkit** (for GPU acceleration)
- **X-Server for Windows** (e.g., [VcXsrv](https://sourceforge.net/projects/vcxsrv/) or [MobaXterm](https://mobaxterm.mobatek.net/)) to view rendering.

### Setup workflow

1. Clone the Google Research Football repository and enter it:

```powershell
git clone https://github.com/google-research/football.git
cd football
```

2. Edit the football `requirements.txt` and `setup.py` file so it uses the compatible dependency versions:

```txt
pygame>=1.9.6
numpy<1.24
```

3. Override the football Dockerfile with the one from this repository:

```powershell
Copy-Item ..\hmarl-grf\Dockerfile .\Dockerfile -Force
```

4. Build the Docker image from the football repository:

Tensorflow without GPU-training support version

```powershell
docker build --build-arg DOCKER_BASE=ubuntu:20.04 . -t gfootball
```

Tensorflow with GPU-training support version

```powershell
docker build --build-arg DOCKER_BASE=tensorflow/tensorflow:1.15.2-gpu-py3 . -t gfootball
```

5. Start the container from this repository with Docker Compose:

```powershell
docker compose up -d
```

6. Open a shell inside the running container:

```powershell
docker exec -it gfootball-dev bash
```

### Rendering
For rendering the game on Windows, ensure your X-Server is running and that Docker has access to the display. If needed, allow Docker containers to connect to the host display:

```powershell
xhost +"local:docker@"
```

This command has to be executed after each reboot. Alternatively, add it to your shell profile to avoid repeating it.

## 📂 Project Structure

- `11v11_ppo.py`: Training script using the Proximal Policy Optimization (PPO) algorithm.
- `11v11_a2c.py`: Training script using the Advantage Actor-Critic (A2C) algorithm.
- `11v11_random_action.py`: A baseline script that executes random actions for all 11 agents.
- `Dockerfile`: Container configuration with all necessary dependencies (Ubuntu 22.04, Python 3.10, GRF, etc.).
- `dumps/`: Directory where environment logs and episode replays are saved.

## 🛠️ Usage

Once inside the container, you can start training by running any of the scripts:

```bash
# Train using PPO
python3 11v11_ppo.py

# Train using A2C
python3 11v11_a2c.py

# Run random action baseline
python3 11v11_random_action.py
```

### Convert a dump file to text
Use the replay dump converter to turn a GRF dump into a readable text trace:

```bash
python3 dumps/dump_to_txt.py \
  --trace_file=/gfootball/dumps/episode_done_20260625-033933885202.dump \
  --output=/gfootball/dumps/output.txt
```

### Plot average team positions
Generate a pitch plot from a dump file:

```bash
python3 evaluation/average_position.py \
  /gfootball/dumps/output.txt \
  /gfootball/dumps/average_position.png
```

### Rendering
If you set `render=True` in the scripts, ensure your X-Server is running on the host and "Disable access control" is checked to allow the container to connect to your display.

## 📝 Environment Details
- **Scenario**: `11_vs_11_stochastic`
- **Representation**: `simple115v2`
- **Agents**: 11 agents controlled on the left team.
- **Action Space**: MultiDiscrete (19 actions per agent).

## 📄 License
This project is part of a thesis repository. Please refer to the specific license terms if applicable.