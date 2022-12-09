import pynini
from nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_NOT_QUOTE,
    GraphFst,
    delete_extra_space,
    delete_preserve_order,
)
from pynini.lib import pynutil


class MoneyFst(GraphFst):
    def __init__(self, decimal: GraphFst, deterministic: bool = True):
        super().__init__(name="money", kind="verbalize", deterministic=deterministic)
        keep_space = pynini.accep(" ")
        maj = pynutil.delete("currency_maj: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        min = pynutil.delete("currency_min: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")

        fractional_part = (
            pynutil.delete("fractional_part: \"") + pynini.closure(NEMO_NOT_QUOTE, 1) + pynutil.delete("\"")
        )

        integer_part = decimal.integer

        #  *** currency_maj
        graph_integer = integer_part + keep_space + maj

        #  *** currency_maj + (***) | ((and) *** current_min)
        fractional = fractional_part + delete_extra_space + min

        graph_integer_with_minor = integer_part + keep_space + maj + keep_space + fractional + delete_preserve_order

        # *** point *** currency_maj
        graph_decimal = decimal.numbers + keep_space + maj

        # *** current_min
        graph_minor = fractional_part + delete_extra_space + min + delete_preserve_order

        graph = graph_integer | graph_integer_with_minor | graph_decimal | graph_minor

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
