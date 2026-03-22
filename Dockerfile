FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt upgrade -y && apt install -y \
    software-properties-common \
    ca-certificates \
    python3 \
    python3-pip \
    python3-dev \
    curl \
    git \
    build-essential \
    cmake \
    autoconf \
    libtool \
    pkg-config \
    flex \
    bison \
    libbz2-dev \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
RUN uv venv --python 3.11

COPY . /app
RUN uv pip install -r /app/requirements.txt

# Verify NLE works
RUN uv run python -c "import nle; import gymnasium as gym; env = gym.make('NetHackScore-v0'); env.reset(); env.close(); print('NLE OK')"

EXPOSE 8080
CMD ["uv", "run", "python", "/app/server.py"]
