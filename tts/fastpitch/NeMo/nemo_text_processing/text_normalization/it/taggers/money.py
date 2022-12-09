import pynini
from fastpitch.NeMo.nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_ALPHA,
    NEMO_DIGIT,
    NEMO_SIGMA,
    SINGULAR_TO_PLURAL,
    GraphFst,
    convert_space,
    insert_space,
)
from fastpitch.NeMo.nemo_text_processing.text_normalization.it.utils import get_abs_path, load_labels
from pynini.lib import pynutil


class MoneyFst(GraphFst):
    """
    Finite state transducer for classifying money, suppletive aware, e.g.
        $12.05 -> money { integer_part: "twelve" currency_maj: "dollars" fractional_part: "five" currency_min: "cents" preserve_order: true }
        $12.0500 -> money { integer_part: "twelve" currency_maj: "dollars" fractional_part: "five" currency_min: "cents" preserve_order: true }
        $1 -> money { currency_maj: "dollar" integer_part: "one" }
        $1.00 -> money { currency_maj: "dollar" integer_part: "one" }
        $0.05 -> money { fractional_part: "five"  currency_min: "cents" preserve_order: true }
        $1 million -> money { currency_maj: "dollars" integer_part: "one" quantity: "million" }
        $1.2 million -> money { currency_maj: "dollars" integer_part: "one"  fractional_part: "two" quantity: "million" }
        $1.2320 -> money { currency_maj: "dollars" integer_part: "one"  fractional_part: "two three two" }

    Args:
        cardinal: CardinalFst
        decimal: DecimalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self, cardinal: GraphFst, decimal: GraphFst, deterministic: bool = True):
        super().__init__(name="money", kind="classify", deterministic=deterministic)

        #graph_integer = decimal.graph_integer
        #graph_fractional = decimal.graph_fractional
        decimal_graph = decimal.final_graph
        cardinal_graph = cardinal.graph

        currency_maj_sing = pynini.string_file(
            get_abs_path("data/money/currency_maj_sing.tsv"))
        currency_maj_plur = pynini.string_file(
            get_abs_path("data/money/currency_maj_plur.tsv"))
        currency_min_sing = pynini.string_file(
            get_abs_path("data/money/currency_min_sing.tsv"))
        currency_min_plur = pynini.string_file(
            get_abs_path("data/money/currency_min_plur.tsv"))

        graph_maj_singular = pynutil.insert(
            "currency_maj: \"") + currency_maj_sing + pynutil.insert("\"")
        graph_maj_plural = pynutil.insert(
            "currency_maj: \"") + currency_maj_plur + pynutil.insert("\"")
        graph_min_singular = pynutil.insert(
            " currency_min: \"") + currency_min_sing + pynutil.insert("\"")
        graph_min_plural = pynutil.insert(
            " currency_min: \"") + currency_min_plur + pynutil.insert("\"")

        graph_integer_one = pynutil.insert(
            "integer_part: \"") + pynini.cross("1", "un") + pynutil.insert("\"")
        graph_fractional_one = pynutil.insert(
            "fractional_part: \"") + pynini.cross("1", "uno") + pynutil.insert("\"")
        virgola = pynutil.delete(",")

        graph_integer = (
            pynutil.insert("integer_part: \"") + ((NEMO_SIGMA - "1")
                                                  @ cardinal_graph) + pynutil.insert("\"")
        )

        graph_fractional = (
            pynutil.insert("fractional_part: \"") + ((NEMO_SIGMA - "1")
                                                     @ cardinal_graph) + pynutil.insert("\"")
        )

        final_graph = (
            graph_integer_one + insert_space + graph_maj_singular
            | decimal_graph + insert_space + graph_maj_plural
            | graph_integer + insert_space + graph_maj_plural
        )

        final_graph = self.add_tokens(final_graph.optimize())
        self.fst = final_graph.optimize()
