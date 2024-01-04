# italian-tts
Italian pre-trained TTS models for VITS, FastPitch and Hifi-GAN.

## Intro
This work was done during my [master thesis](https://webthesis.biblio.polito.it/25614/).

This is a demo of the Italian TTS models.
You write the text you want to synthesize in a txt file, while the speech is automatically saved in `output.mp3`.
The pre-trained checkpoints are available in the `checkpoints` folder.

## Settings
The settings must be written in a `.env` file that will be automatically read by docker
when running `docker compose up`.
You must set:
* `TEXT_FILE`: path to the text file
* `MODEL`: the TTS model to use for inference. Can be one of `vits`, `fp_male`, `fp_female`.

Example of `.env`:

```
TEXT_FILE=text.txt
MODEL=fp_male
```

## How to run

if first_time:

```
git clone https://github.com/simsax/italian-tts.git
cd italian-tts
# create .env and the text file
sudo chmod +x init.sh
./init.sh
```

else:
```
# create .env and the text file
docker compose up
```

## References
* https://github.com/NVIDIA/NeMo
* https://github.com/jaywalnut310/vits
* https://www.caito.de/2019/01/03/the-m-ailabs-speech-dataset/
* https://www.liberliber.it/online/autori/autori-d/charles-dickens/le-avventure-di-nicola-nickleby-audiolibro/
