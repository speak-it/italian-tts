import pynini
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space
from pynini.lib import pynutil


class OrdinalFst(GraphFst):
    """
    Finite state transducer for verbalizing ordinal, e.g.
        ordinal { integer: "primo" } -> primo

    """

    def __init__(self):
        super().__init__(name="ordinal", kind="verbalize")

        integer = pynini.closure(NEMO_NOT_QUOTE)

        self.integer = delete_space + pynutil.delete("\"") + integer + pynutil.delete("\"")
        integer = pynutil.delete("integer:") + self.integer

        delete_tokens = self.delete_tokens(integer)
        self.fst = delete_tokens.optimize()