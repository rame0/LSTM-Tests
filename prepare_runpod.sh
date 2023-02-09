#!/usr/bin/env bash

apt udpate
apt upgrade -y

apt install --reinstall python3-pip -y
pip3 install --upgrade pip

pip3 install numpy, scikit-learn, pandas