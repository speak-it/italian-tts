import pynini
from fastpitch.NeMo.nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_ALPHA,
    NEMO_DIGIT,
    NEMO_NON_BREAKING_SPACE,
    NEMO_SIGMA,
    NEMO_SPACE,
    NEMO_UPPER,
    SINGULAR_TO_PLURAL,
    TO_LOWER,
    GraphFst,
    convert_space,
    delete_space,
    delete_zero_or_one_space,
    insert_space,
)
from fastpitch.NeMo.nemo_text_processing.text_normalization.it.taggers.ordinal import OrdinalFst as OrdinalTagger
from fastpitch.NeMo.nemo_text_processing.text_normalization.it.utils import get_abs_path, load_labels
from fastpitch.NeMo.nemo_text_processing.text_normalization.it.verbalizers.ordinal import OrdinalFst as OrdinalVerbalizer
from pynini.examples import plurals
from pynini.lib import pynutil


class MeasureFst(GraphFst):
    """
    Finite state transducer for classifying measure, suppletive aware, e.g.
        -12kg -> measure { negative: "true" cardinal { integer: "twelve" } units: "kilograms" }
        1kg -> measure { cardinal { integer: "one" } units: "kilogram" }
        .5kg -> measure { decimal { fractional_part: "five" } units: "kilograms" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, deterministic: bool = True):
        super().__init__(name="measure", kind="classify", deterministic=deterministic)
        cardinal_graph = cardinal.graph

        graph_unit_sing = pynini.string_file(
            get_abs_path("data/measure/unit_singular.tsv"))

        graph_unit_sing |= pynini.compose(
            pynini.closure(TO_LOWER, 1) + (NEMO_ALPHA | TO_LOWER) +
            pynini.closure(NEMO_ALPHA | TO_LOWER), graph_unit_sing
        ).optimize()

        graph_unit_plural = pynini.string_file(
            get_abs_path("data/measure/unit_plural.tsv"))
        graph_unit_plural |= pynini.compose(
            pynini.closure(TO_LOWER, 1) + (NEMO_ALPHA | TO_LOWER) +
            pynini.closure(NEMO_ALPHA | TO_LOWER), graph_unit_plural
        ).optimize()

        remove_apostrofo = pynini.string_map([
            ("quell'", "quell "),
            ("dell'", "dell "),
            ("l'", "l "),
            ("all'", "all "),
            ("dall'", "dall ")
        ])

        optional_graph_negative = pynini.closure(pynutil.insert(
            "negative: ") + pynini.cross("-", "\"true\" "), 0, 1)
        optional_graph_positive = pynini.closure(pynutil.insert(
            "positive: ") + pynini.cross("+", "\"true\" "), 0, 1)
        optional_words = pynini.closure(pynutil.insert("opt_word: \"") +
                                        remove_apostrofo + pynutil.insert("\" "), 0, 1)

        unit_plural = (
            pynutil.insert("units: \"")
            + graph_unit_plural
            + pynutil.insert("\"")
        )

        unit_singular = (
            pynutil.insert("units: \"") + graph_unit_sing +
            pynutil.insert("\"")
        )

        subgraph_decimal = (
            pynutil.insert("decimal { ")
            + optional_words
            + optional_graph_positive
            + optional_graph_negative
            + decimal.final_graph_wo_negative
            + delete_space
            + pynutil.insert(" } ")
            + unit_plural
        )

        subgraph_cardinal = (
            pynutil.insert("cardinal { ")
            + optional_words
            + optional_graph_positive
            + optional_graph_negative
            + pynutil.insert("integer: \"")
            + ((NEMO_SIGMA - "1") @ cardinal_graph)
            + delete_space
            + pynutil.insert("\"")
            + pynutil.insert(" } ")
            + unit_plural
        )

        subgraph_cardinal |= (
            pynutil.insert("cardinal { ")
            + optional_words
            + optional_graph_positive
            + optional_graph_negative
            + pynutil.insert("integer: \"")
            + pynini.cross("1", "un")
            + delete_space
            + pynutil.insert("\"")
            + pynutil.insert(" } ")
            + unit_singular
        )

        unit_graph = (
            pynutil.insert("cardinal { integer: \"-\" } units: \"")
            + delete_zero_or_one_space
            + pynutil.insert(NEMO_NON_BREAKING_SPACE)
            + graph_unit_sing
            + pynutil.insert("\" preserve_order: true")
        )

        final_graph = (
            subgraph_decimal
            | subgraph_cardinal
            | unit_graph
        )

        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
