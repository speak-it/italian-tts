import pynini
from fastpitch.NeMo.nemo_text_processing.text_normalization.en.graph_utils import GraphFst, convert_space
from fastpitch.NeMo.nemo_text_processing.text_normalization.it.utils import get_abs_path, load_labels
from pynini.lib import pynutil


class WhiteListFst(GraphFst):
    """
    Finite state transducer for classifying whitelist, e.g.
        "sr." -> tokens { name: "señor" }
    This class has highest priority among all classifier grammars. Whitelisted tokens are defined and loaded from "data/whitelist.tsv".

    Args:
        input_case: accepting either "lower_cased" or "cased" input.
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
        input_file: path to a file with whitelist replacements
    """

    def __init__(self, input_case: str, input_file: str = None):
        super().__init__(name="whitelist", kind="classify")

        def _get_whitelist_graph(input_case, file):
            whitelist = load_labels(file)
            if input_case == "lower_cased":
                whitelist = [[x[0].lower()] + x[1:] for x in whitelist]
            graph = pynini.string_map(whitelist)
            return graph

        graph = _get_whitelist_graph(
            input_case, get_abs_path("data/whitelist.tsv"))

        if input_file:
            whitelist_provided = _get_whitelist_graph(input_case, input_file)
            graph = whitelist_provided

        self.graph = graph
        self.final_graph = convert_space(self.graph).optimize()
        self.fst = (pynutil.insert("name: \"") +
                    self.final_graph + pynutil.insert("\"")).optimize()
