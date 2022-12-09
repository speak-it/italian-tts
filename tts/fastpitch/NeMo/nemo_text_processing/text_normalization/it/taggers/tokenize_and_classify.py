import os

import pynini
from fastpitch.NeMo.nemo_text_processing.text_normalization.en.graph_utils import (
    NEMO_WHITE_SPACE,
    GraphFst,
    delete_extra_space,
    delete_space,
    generator_main,
)

from fastpitch.NeMo.nemo_text_processing.text_normalization.it.taggers.cardinal import CardinalFst
from fastpitch.NeMo.nemo_text_processing.text_normalization.it.taggers.decimal import DecimalFst
from fastpitch.NeMo.nemo_text_processing.text_normalization.it.taggers.ordinal import OrdinalFst
from fastpitch.NeMo.nemo_text_processing.text_normalization.it.taggers.whitelist import WhiteListFst
from fastpitch.NeMo.nemo_text_processing.text_normalization.it.taggers.money import MoneyFst
from fastpitch.NeMo.nemo_text_processing.text_normalization.it.taggers.measure import MeasureFst
from fastpitch.NeMo.nemo_text_processing.text_normalization.it.taggers.time import TimeFst
from fastpitch.NeMo.nemo_text_processing.text_normalization.it.taggers.electronic import ElectronicFst
from fastpitch.NeMo.nemo_text_processing.text_normalization.it.taggers.abbreviation import AbbreviationFst
from fastpitch.NeMo.nemo_text_processing.text_normalization.en.taggers.punctuation import PunctuationFst
from fastpitch.NeMo.nemo_text_processing.text_normalization.es.taggers.word import WordFst
from pynini.lib import pynutil

from nemo.utils import logging


class ClassifyFst(GraphFst):
    """
    Final class that composes all other classification grammars. This class can process an entire sentence, that is lower cased.
    For deployment, this grammar will be compiled and exported to OpenFst Finate State aRchive (FAR) File.
    More details to deployment at NeMo/tools/text_processing_deployment.

    Args:
        input_case: accepting either "lower_cased" or "cased" input.
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
        cache_dir: path to a dir with .far grammar file. Set to None to avoid using cache.
        overwrite_cache: set to True to overwrite .far files
        whitelist: path to a file with whitelist replacements
    """

    def __init__(
        self,
        input_case: str,
        deterministic: bool = False,
        cache_dir: str = None,
        overwrite_cache: bool = False,
        whitelist: str = None,
    ):
        super().__init__(name="tokenize_and_classify",
                         kind="classify", deterministic=deterministic)
        far_file = None
        if cache_dir is not None and cache_dir != "None":
            os.makedirs(cache_dir, exist_ok=True)
            whitelist_file = os.path.basename(whitelist) if whitelist else ""
            far_file = os.path.join(
                cache_dir, f"_{input_case}_it_tn_{deterministic}_deterministic{whitelist_file}.far"
            )
        if not overwrite_cache and far_file and os.path.exists(far_file):
            self.fst = pynini.Far(far_file, mode="r")["tokenize_and_classify"]
            logging.info(f"ClassifyFst.fst was restored from {far_file}.")
        else:
            logging.info(
                f"Creating ClassifyFst grammars. This might take some time...")

            self.cardinal = CardinalFst()
            cardinal_graph = self.cardinal.fst

            self.ordinal = OrdinalFst()
            ordinal_graph = self.ordinal.fst

            self.decimal = DecimalFst(cardinal=self.cardinal)
            decimal_graph = self.decimal.fst

            word_graph = WordFst(deterministic=deterministic).fst

            self.whitelist = WhiteListFst(
                input_case=input_case, input_file=whitelist)
            whitelist_graph = self.whitelist.fst

            self.money = MoneyFst(cardinal=self.cardinal, decimal=self.decimal)
            money_graph = self.money.fst

            self.measure = MeasureFst(
                cardinal=self.cardinal, decimal=self.decimal)
            measure_graph = self.measure.fst

            self.time = TimeFst(cardinal=self.cardinal)
            time_graph = self.time.fst

            self.electronic = ElectronicFst()
            electronic_graph = self.electronic.fst

            self.abbreviation = AbbreviationFst(self.whitelist)
            abbreviation_graph = self.abbreviation.fst

            punct_graph = PunctuationFst(deterministic=deterministic).fst

            classify = (
                pynutil.add_weight(whitelist_graph, 1.01)
                | pynutil.add_weight(cardinal_graph, 1.2)
                | pynutil.add_weight(ordinal_graph, 1.1)
                | pynutil.add_weight(decimal_graph, 1.1)
                | pynutil.add_weight(money_graph, 1.05)
                | pynutil.add_weight(measure_graph, 1.04)
                | pynutil.add_weight(time_graph, 1.05)
                | pynutil.add_weight(electronic_graph, 1.05)
                | pynutil.add_weight(word_graph, 200)
                | pynutil.add_weight(abbreviation_graph, 1.05)
            )
            punct = pynutil.insert(
                "tokens { ") + pynutil.add_weight(punct_graph, weight=2.1) + pynutil.insert(" }")
            punct = pynini.closure(
                pynini.compose(pynini.closure(
                    NEMO_WHITE_SPACE, 1), delete_extra_space)
                | (pynutil.insert(" ") + punct),
                1,
            )
            token = pynutil.insert(
                "tokens { ") + classify + pynutil.insert(" }")
            token_plus_punct = (
                pynini.closure(punct + pynutil.insert(" ")) +
                token + pynini.closure(pynutil.insert(" ") + punct)
            )

            graph = token_plus_punct + pynini.closure(
                (
                    pynini.compose(pynini.closure(
                        NEMO_WHITE_SPACE, 1), delete_extra_space)
                    | (pynutil.insert(" ") + punct + pynutil.insert(" "))
                )
                + token_plus_punct
            )

            graph = delete_space + graph + delete_space
            graph |= punct

            self.fst = graph.optimize()

            if far_file:
                generator_main(far_file, {"tokenize_and_classify": self.fst})
                logging.info(f"ClassifyFst grammars are saved to {far_file}.")
