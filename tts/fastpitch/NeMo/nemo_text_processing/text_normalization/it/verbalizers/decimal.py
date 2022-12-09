import pynini
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space, insert_space
from pynini.lib import pynutil


class DecimalFst(GraphFst):
    """
    Finite state transducer for verbalizing decimal, e.g.
        decimal { negative: "true" integer_part: "twelve" fractional_part: "five o o six" quantity: "billion" } -> minus twelve point five o o six billion

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self):
        super().__init__(name="decimal", kind="verbalize")
        self.optional_negative = pynini.cross("negative: \"true\"", "meno ")
        self.optional_negative = pynini.closure(self.optional_negative + delete_space, 0, 1)
        self.optional_positive = pynini.cross("positive: \"true\"", "più ")
        self.optional_positive = pynini.closure(self.optional_positive + delete_space, 0, 1)
        integer = pynini.closure(NEMO_NOT_QUOTE)
        integer = delete_space + pynutil.delete("\"") + integer + pynutil.delete("\"")
        self.integer = pynutil.delete("integer_part:") + integer
        self.optional_integer = pynini.closure(self.integer + delete_space + insert_space, 0, 1)
        
        self.fractional_default = (
            pynutil.delete("fractional_part:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        self.fractional = pynutil.insert("virgola ") + self.fractional_default

        self.quantity = (
            delete_space
            + insert_space
            + pynutil.delete("quantity:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        self.optional_quantity = pynini.closure(self.quantity, 0, 1)

        graph = self.optional_positive + self.optional_negative + (
            self.integer
            | (self.integer + self.quantity)
            | (self.optional_integer + self.fractional + self.optional_quantity)
        )

        self.numbers = graph
        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()