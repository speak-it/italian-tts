import torch
from vits import commons
from vits import utils
from vits.models import SynthesizerTrn
from vits.text.symbols import symbols
from vits.text import text_to_sequence
from model_interface import Model


class VitsModel(Model):
    def __init__(self, checkpoint_path):
        config = f"./vits/configs/ljs_base.json"
        self.hps = utils.get_hparams_from_file(config)

        self.net_g = SynthesizerTrn(
            len(symbols),
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            **self.hps.model)
        _ = self.net_g.eval()

        _ = utils.load_checkpoint(checkpoint_path, self.net_g, None)

    def get_text(self, text, hps):
        text_norm = text_to_sequence(text, hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    @torch.inference_mode()
    def synthesize(self, text):
        stn_tst = self.get_text(text, self.hps)

        with torch.no_grad():
            x_tst = stn_tst.unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)])
            audio = self.net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[
                0][0, 0].data.float().numpy()
        return audio
