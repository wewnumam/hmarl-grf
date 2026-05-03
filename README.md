# HMARL-GRF: Hybrid Multi-Agent Reinforcement Learning for Google Research Football

This repository contains tools and scripts for training multi-agent reinforcement learning models in the [Google Research Football (GRF)](https://github.com/google-research/football) environment. It focuses on 11v11 scenarios using algorithms like PPO and A2C via Stable Baselines3.

## 🚀 Quick Start

### Prerequisites
- **Docker Desktop** (with WSL2 backend recommended)
- **NVIDIA Container Toolkit** (for GPU acceleration)
- **X-Server for Windows** (e.g., [VcXsrv](https://sourceforge.net/projects/vcxsrv/) or [MobaXterm](https://mobaxterm.mobatek.net/)) to view rendering.

### Running in Docker
In order to build Docker image you have to checkout GFootball git repository first:

```powershell
git clone https://github.com/google-research/football.git
cd football
```

For rendering the game on macOS and Windows, we recommend installing the game according to the instructions for your platform in README.

#### Configure Docker for Rendering
In order to see rendered game you need to allow Docker containers access X server:

```powershell
xhost +"local:docker@"
```

This command has to be executed after each reboot. Alternatively you can add this command to /etc/profile to not worry about it in the future.

#### Build Docker image
Tensorflow without GPU-training support version

```powershell
docker build --build-arg DOCKER_BASE=ubuntu:20.04 . -t gfootball
```

Tensorflow with GPU-training support version

```powershell
docker build --build-arg DOCKER_BASE=tensorflow/tensorflow:1.15.2-gpu-py3 . -t gfootball
```

### Building the Image

Build the Docker image locally:

```powershell
docker build -t gfootball .
```

### Running the Container

To run the project with live file synchronization (so changes on your host reflect inside the container):

**PowerShell:**

```powershell
docker run --gpus all `
  -e DISPLAY=host.docker.internal:0.0 `
  -it `
  -v ${PWD}:/gfootball `
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw `
  gfootball bash
```

**Linux/WSL:**

```bash
docker run --gpus all \
  -e DISPLAY=$DISPLAY \
  -it \
  -v $(pwd):/gfootball \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  gfootball bash
```

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

### Rendering
If you set `render=True` in the scripts, ensure your X-Server is running on the host and "Disable access control" is checked to allow the container to connect to your display.

## 📝 Environment Details
- **Scenario**: `11_vs_11_stochastic`
- **Representation**: `simple115v2`
- **Agents**: 11 agents controlled on the left team.
- **Action Space**: MultiDiscrete (19 actions per agent).

## 📄 License
This project is part of a thesis repository. Please refer to the specific license terms if applicable.