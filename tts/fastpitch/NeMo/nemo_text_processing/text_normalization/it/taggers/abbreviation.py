import pynini
from fastpitch.NeMo.nemo_text_processing.text_normalization.en.graph_utils import NEMO_UPPER, GraphFst, insert_space
from fastpitch.NeMo.nemo_text_processing.text_normalization.it.utils import get_abs_path

from pynini.lib import pynutil


class AbbreviationFst(GraphFst):
    """
    Finite state transducer for classifying electronic: as URLs, email addresses, etc.
        e.g. "ABC" -> tokens { abbreviation { value: "A B C" } }
    """

    def __init__(self, whitelist: 'pynini.FstLike', deterministic: bool = True):
        super().__init__(name="abbreviation", kind="classify", deterministic=deterministic)

        letters_map = pynini.string_file(
            get_abs_path("data/abbreviation/letters.tsv"))

        dot = pynutil.delete(".")

        # A.B.C. -> A. B. C.
        graph = letters_map + dot + \
            pynini.closure(insert_space + letters_map + dot, 1)
        # A.B.C. -> A.B.C.
        graph |= letters_map + dot + pynini.closure(letters_map + dot, 1)
        # ABC -> A B C
        graph |= letters_map + pynini.closure(insert_space + letters_map, 1)

        # exclude words that are included in the whitelist
        graph = pynini.compose(
            pynini.difference(pynini.project(graph, "input"),
                              pynini.project(whitelist.graph, "input")), graph
        )

        graph = pynutil.insert("value: \"") + \
            graph.optimize() + pynutil.insert("\"")
        graph = self.add_tokens(graph)
        self.fst = graph.optimize()
