#!/usr/bin/env bash
chmod 1777 /tmp
apt update && apt-get install cmake -y && apt-get install sox -y
apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3.10-venv \
    build-essential \
    netcat \
    git \
    && rm -rf /var/lib/apt/lists/*

ln -s /usr/bin/python3 /usr/bin/python
pip3 install --upgrade pip
echo << python --version
cd tools
./setup_venv.sh $(command -v python3)

bash activate_python.sh

make





