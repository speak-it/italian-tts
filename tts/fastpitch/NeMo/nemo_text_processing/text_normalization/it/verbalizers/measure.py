import pynini
from fastpitch.NeMo.nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space, insert_space
from pynini.lib import pynutil


class MeasureFst(GraphFst):
    """
    Finite state transducer for verbalizing measure, e.g.
        measure { negative: "true" cardinal { integer: "twelve" } units: "kilograms" } -> minus twelve kilograms
        measure { decimal { integer_part: "twelve" fractional_part: "five" } units: "kilograms" } -> twelve point five kilograms
        tokens { measure { units: "covid" decimal { integer_part: "nineteen"  fractional_part: "five" }  } } -> covid nineteen point five

    Args:
        decimal: DecimalFst
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, decimal: GraphFst, cardinal: GraphFst, deterministic: bool = True):
        super().__init__(name="measure", kind="verbalize", deterministic=deterministic)
        optional_positive = cardinal.optional_positive
        optional_negative = cardinal.optional_negative
        optional_word = pynini.closure(
            pynutil.delete("opt_word: \"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
            + delete_space, 0, 1)

        unit = (
            pynutil.delete("units: \"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
            + delete_space
        )

        graph_decimal = (
            pynutil.delete("decimal {")
            + delete_space
            + optional_word
            + delete_space
            + optional_positive
            + delete_space
            + optional_negative
            + delete_space
            + decimal.numbers
            + delete_space
            + pynutil.delete("}")
        )
        graph_cardinal = (
            pynutil.delete("cardinal {")
            + delete_space
            + optional_word
            + delete_space
            + optional_positive
            + delete_space
            + optional_negative
            + delete_space
            + cardinal.numbers
            + delete_space
            + pynutil.delete("}")
        )

        graph = (graph_cardinal | graph_decimal) + \
            delete_space + insert_space + unit

        # SH adds "preserve_order: true" by default
        preserve_order = pynutil.delete(
            "preserve_order:") + delete_space + pynutil.delete("true") + delete_space
        graph |= unit + insert_space + \
            (graph_cardinal | graph_decimal) + \
            delete_space + pynini.closure(preserve_order)
        # for only unit
        graph |= (
            pynutil.delete("cardinal { integer: \"-\"")
            + delete_space
            + pynutil.delete("}")
            + delete_space
            + unit
            + pynini.closure(preserve_order)
        )

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
