ARG DOCKER_BASE=ubuntu:22.04
FROM ${DOCKER_BASE}

ENV DEBIAN_FRONTEND=noninteractive

RUN rm -f /etc/apt/sources.list.d/cuda*.list || true \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
    git cmake build-essential \
    libgl1-mesa-dev libsdl2-dev \
    libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev \
    libboost-all-dev \
    libdirectfb-dev \
    mesa-utils xvfb x11vnc \
    python3-pip \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel
RUN python3 -m pip install psutil

COPY . /gfootball
RUN cd /gfootball && python3 -m pip install .

WORKDIR /gfootball