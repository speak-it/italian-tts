import pynini
import string
from pynini.lib import pynutil

from fastpitch.NeMo.nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_ALPHA,
    NEMO_DIGIT,
    NEMO_SIGMA,
    NEMO_SPACE,
    NEMO_WHITE_SPACE,
    GraphFst,
    delete_space,
    insert_space,
)

from fastpitch.NeMo.nemo_text_processing.text_normalization.it.utils import get_abs_path


class DecimalFst(GraphFst):
    def __init__(self, cardinal: GraphFst):
        super().__init__(name="decimal", kind="classify")

        cardinal_graph = cardinal.graph

        virgola = pynutil.delete(",")
        optional_graph_negative = pynini.closure(pynutil.insert(
            "negative: ") + pynini.cross("-", "\"true\" "), 0, 1)
        optional_graph_positive = pynini.closure(pynutil.insert(
            "positive: ") + pynini.cross("+", "\"true\" "), 0, 1)

        self.graph_fractional = pynutil.insert(
            "fractional_part: \"") + cardinal.cifre + pynutil.insert("\"")
        self.graph_integer = pynutil.insert(
            "integer_part: \"") + cardinal_graph + pynutil.insert("\"")

        self.final_graph_wo_negative = (
            pynini.closure(self.graph_integer + pynutil.insert(" "), 0, 1)
            + virgola
            + pynutil.insert(" ")
            + self.graph_fractional
        )

        self.final_graph = (
            optional_graph_positive
            + optional_graph_negative
            + self.final_graph_wo_negative
        )

        final_graph = self.add_tokens(self.final_graph)
        self.fst = final_graph.optimize()
