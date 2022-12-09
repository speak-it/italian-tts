import pynini
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space
from pynini.lib import pynutil


class CardinalFst(GraphFst):
    """
    Finite state transducer for verbalizing cardinal, e.g.
        cardinal { negative: "true" integer: "23" } -> meno venti tre

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="cardinal", kind="verbalize", deterministic=deterministic)

        self.optional_negative = pynini.cross("negative: \"true\"", "meno ")
        self.optional_negative = pynini.closure(self.optional_negative + delete_space, 0, 1)

        self.optional_positive = pynini.cross("positive: \"true\"", "più ")
        self.optional_positive = pynini.closure(self.optional_positive + delete_space, 0, 1)

        integer = pynini.closure(NEMO_NOT_QUOTE)

        self.integer = delete_space + pynutil.delete("\"") + integer + pynutil.delete("\"")
        integer = pynutil.delete("integer:") + self.integer

        self.numbers = self.optional_positive + self.optional_negative + integer
        delete_tokens = self.delete_tokens(self.numbers)
        self.fst = delete_tokens.optimize()