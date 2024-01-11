import numpy as np
from pydub import AudioSegment, effects


class PodcastGenerator:
    def __init__(self, jingle_path="jingles/default_jingle.mp3"):
        self.__jingle = AudioSegment.from_mp3(jingle_path)

    @property
    def jingle(self):
        return self.__jingle

    @jingle.setter
    def jingle(self, jingle_path):
        self.__jingle = AudioSegment.from_mp3(jingle_path)

    def generate_segment(self, audio_segments):
        final_segment = AudioSegment.empty()
        final_segment += self.__jingle
        for audio_segment in audio_segments:
            final_segment += audio_segment
            final_segment += self.__jingle

        return effects.normalize(final_segment)
