import pynini
from nemo_text_processing.text_normalization.en.graph_utils import NEMO_CHAR, NEMO_SIGMA, GraphFst, delete_space
from pynini.lib import pynutil


class WhiteListFst(GraphFst):
    """
    Finite state transducer for verbalizing whitelist
        e.g. tokens { name: "misses" } } -> misses

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="whitelist", kind="verbalize", deterministic=deterministic)
        graph = (
            pynutil.delete("name:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_CHAR - " ", 1)
            + pynutil.delete("\"")
        )
        graph = graph @ pynini.cdrewrite(pynini.cross(u"\u00A0", " "), "", "", NEMO_SIGMA)
        self.fst = graph.optimize()