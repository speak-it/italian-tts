import os
from pydub import AudioSegment
from synthesizer import Synthesizer
from fastpitch.tts_model import FastpitchModel
from vits.tts_model import VitsModel
from logger import get_logger

log = get_logger(__name__)


def save_as_mp3(audio_path, audio):
    audio_segment = AudioSegment(
        audio.tobytes(),
        frame_rate=16000,
        sample_width=audio.dtype.itemsize,
        channels=1
    )
    audio_segment.export(audio_path, format="mp3", bitrate="160k")


def create_synthesizer():
    model = os.getenv("MODEL")
    BASE = "/checkpoints"
    if model == "vits":
        vits_model = VitsModel(f"{BASE}/vits/vits.pth")
        return Synthesizer(vits_model)
    elif model == "fp_male":
        male1_model = FastpitchModel(
            f"{BASE}/fastpitch/male1/FastPitch.ckpt", f"{BASE}/fastpitch/male1/HifiGan.ckpt",
            "./male_conf.yaml")
        return Synthesizer(male1_model)
    elif model == "fp_female":
        female1_model = FastpitchModel(
            f"{BASE}/fastpitch/female1/FastPitch.ckpt", f"{BASE}/fastpitch/female1/HifiGan.ckpt",
            "./female_conf.yaml")
        return Synthesizer(female1_model)
    else:
        log.error("Model not found.")
        exit(1)


if __name__ == "__main__":
    text_file = f"/local/{os.getenv('TEXT_FILE')}"
    if not os.path.isfile(text_file):
        log.error("Missing path to txt file.")
        exit(1)

    else:
        with open(text_file, "r") as f:
            text = f.read()

    synthesizer = create_synthesizer()
    audio = synthesizer.text_to_speech(text)
    save_as_mp3("/local/output.mp3", audio)
    log.info("Audio file saved as output.mp3.")
