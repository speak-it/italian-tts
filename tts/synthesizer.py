import re
import numpy as np
from pydub import AudioSegment
from pydub.silence import detect_silence
import io
import soundfile as sf
from model_interface import Model


class Synthesizer:
    def __init__(self, model: Model):
        self.__model = model

    def text_to_speech(self, text):
        audios = []
        sentences = re.findall(r'.*?[.!:();\?]|.+?$', text)
        max_n_words = 25
        for sentence in sentences:
            n_words = len(sentence.split(" "))
            if n_words > max_n_words:  # split
                shorter_sentences = self.__split_by_words(
                    sentence, max_n_words)
                for shorter_sentence in shorter_sentences:
                    text = self.__add_padding(shorter_sentence)
                    audio = self.__cut_padding(self.__model.synthesize(text))
                    audios.append(audio)
            else:
                text = self.__add_padding(sentence)
                audios.append(self.__cut_padding(
                    self.__model.synthesize(text)))
        return np.concatenate(audios)

    def __split_by_words(self, text, n_words):
        words = text.split()
        sentences = [' '.join(words[i: i + n_words])
                     for i in range(0, len(words), n_words)]
        return sentences

    def __add_padding(self, text):
        return "prima. " + text + ". prima."

    def __cut_padding(self, audio):
        tmp_file = io.BytesIO()
        sf.write(tmp_file, audio, samplerate=16000, format='wav')
        audio_segment = AudioSegment.from_wav(tmp_file)
        silences = detect_silence(
            audio_segment, min_silence_len=50, silence_thresh=-30, seek_step=1)
        silence_pad = 50  # ms
        start = silences[1][1] - silence_pad
        end = silences[-2][0] + silence_pad
        audio_segment = audio_segment[start:end]
        return np.array(audio_segment.get_array_of_samples())
