# speak-it

A platform that creates podcasts of news articles using Text-to-Speech.

## Settings
The settings for the API must be written in a `.env` file that will be automatically read by Docker
when running `docker compose up`.

Example of `.env`:

```
# fastapi service
PORT=5050
WEB_CONCURRENCY=1 # number of uvicorn workers

# tts service
TTS_PROCESSES=1 # number of tts processes
TORCH_THREADS=2 # number of PyTorch threads for each tts process
MODEL=FastPitch # model to use (FastPitch/Vits)
TTS_MAX_CPUS=2  # max number of cores for this service
```

## How to run

```
git clone git@github.com:simsax/italian-tts.git
cd italian-tts

# create .env file as specified above

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
```

## Note
* The tts service finishes loading when it logs `TTS model loaded. Ready to consume.`.
* Take a look at http://localhost:5050/docs.

## References
* https://github.com/NVIDIA/NeMo
* https://github.com/jaywalnut310/vits
