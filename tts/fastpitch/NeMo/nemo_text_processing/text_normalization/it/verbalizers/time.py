import pynini
from fastpitch.NeMo.nemo_text_processing.text_normalization.en.graph_utils import NEMO_NOT_QUOTE, GraphFst, delete_space, insert_space
from fastpitch.NeMo.nemo_text_processing.text_normalization.it.utils import get_abs_path
from pynini.lib import pynutil
import string


class TimeFst(GraphFst):
    def __init__(self):
        super().__init__(name="time", kind="verbalize")

        letters = pynini.union(*string.ascii_lowercase)
        sigma_star = pynini.union(letters, ' ').closure().optimize()
        hour_mappings = pynini.string_file(get_abs_path("data/time/hours.tsv"))
        min_mappings = pynini.string_file(
            get_abs_path("data/time/minutes.tsv"))

        exceptions = (
            pynini.cdrewrite(pynutil.delete("zero"), "[BOS]", " ", sigma_star)
            @ pynini.cdrewrite(hour_mappings, "[BOS]", " e", sigma_star)
            @ pynini.cdrewrite(pynutil.delete(" e zero zero"), "", "", sigma_star)
            @ pynini.cdrewrite(min_mappings, "", "", sigma_star)
        ).optimize()

        self.hours = (
            pynutil.delete("hours:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        self.minutes = (
            delete_space
            + pynutil.delete("minutes:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )

        self.suffix = (
            delete_space
            + insert_space
            + pynutil.delete("suffix:")
            + delete_space
            + pynutil.delete("\"")
            + pynini.closure(NEMO_NOT_QUOTE, 1)
            + pynutil.delete("\"")
        )
        self.optional_suffix = pynini.closure(self.suffix, 0, 1)

        graph = self.hours + \
            pynutil.insert(" e ") + self.minutes + self.optional_suffix
        graph = graph @ exceptions

        delete_tokens = self.delete_tokens(graph)
        self.fst = delete_tokens.optimize()
