#!/bin/bash

# required for downloading gdrive files
pip install gdown

# download checkpoints
gdown --folder https://drive.google.com/drive/folders/1GYx7vhNi07DClXrzLDgau_LV-aHD2-yz

# copy checkpoints into docker volume
cd checkpoints
docker volume create checkpoints
docker container create --name temp -v checkpoints:/data busybox
docker cp . temp:/data
docker rm temp
cd ..

# run containers
docker compose up --build
