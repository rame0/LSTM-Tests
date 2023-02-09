#!/usr/bin/env bash

apt update
apt upgrade -y
apt install -y vim tmux htop

apt install --reinstall python3-pip -y
pip3 install --upgrade pip

pip3 install numpy scikit-learn pandas