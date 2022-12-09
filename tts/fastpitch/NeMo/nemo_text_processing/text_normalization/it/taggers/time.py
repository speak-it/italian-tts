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


class TimeFst(GraphFst):
    def __init__(self, cardinal: GraphFst):
        super().__init__(name="time", kind="classify")
        cardinal_graph = cardinal.graph
        colon = pynutil.delete(":")
        hour = cardinal_graph + colon + pynutil.delete(cardinal_graph)
        minutes = cardinal_graph + colon + pynutil.delete(cardinal_graph)

        graph_hour = pynutil.insert("hours: \"") + \
            cardinal_graph + pynutil.insert("\"")
        graph_min = pynutil.insert("minutes: \"") + \
            cardinal_graph + pynutil.insert("\"")

        final_graph = (
            graph_hour
            + colon
            + insert_space
            + graph_min
        )

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
