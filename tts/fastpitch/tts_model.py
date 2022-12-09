from fastpitch.NeMo.nemo.collections.tts.models import HifiGanModel
from fastpitch.NeMo.nemo.collections.tts.models import FastPitchModel
from fastpitch.NeMo.nemo_text_processing.text_normalization.normalize import Normalizer
import numpy as np
from model_interface import Model
import torch
from omegaconf import OmegaConf


class FastpitchModel(Model):
    def __init__(self, spec_gen_path: str, vocoder_path: str, conf_path: str):
        # hack to make it work
        conf = OmegaConf.load(conf_path)
        self.__spec_gen = FastPitchModel(cfg=conf.model)
        self.__spec_gen.maybe_init_from_pretrained_checkpoint(cfg=conf)
        self.__spec_gen.freeze()
        self.__spec_gen.eval()
        # load spectrogram generator and vocoder checkpoints
        # self.__spec_gen = FastPitchModel.load_from_checkpoint(checkpoint_path=spec_gen_path,
        #                                                       hparams_file="/app/fast_align.yaml").eval()
        self.__vocoder = HifiGanModel.load_from_checkpoint(
            checkpoint_path=vocoder_path).eval()

        # change to italian normalizer
        self.__spec_gen.normalizer = Normalizer(lang="it", input_case="cased")
        self.__spec_gen.text_normalizer_call = self.__spec_gen.normalizer.normalize

    @torch.inference_mode()
    def synthesize(self, text):
        parsed = self.__spec_gen.parse(str_input=text)
        spectrogram = self.__spec_gen.generate_spectrogram(tokens=parsed)
        audio = self.__vocoder.convert_spectrogram_to_audio(spec=spectrogram)

        audio = audio.to('cpu').detach().numpy()[0]
        audio = audio / np.abs(audio).max()
        return audio
