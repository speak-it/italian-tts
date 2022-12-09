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
from pynini.lib import pynutil


class OrdinalFst(GraphFst):
    """
    Finite state transducer for classifying ordinal, e.g.
        13º -> ordinal { integer: "tredicesimo" }

    Args:
        cardinal: CardinalFst
        deterministic: if True will provide a single transduction option,
            for False multiple transduction are generated (used for audio-based normalization)
    """

    def __init__(self):
        super().__init__(name="ordinal", kind="classify")

        _digit = pynini.union(*string.digits)  # 0 | 1 | ... | 9
        # Powers of ten that have single-word representations in English. E1*
        # is a special symbol we will use in the teens below.
        _powers = pynini.union("[E1]", "[E1*]", "[E2]", "[E3]", "[E6]", "[E9]")
        ordinal = pynini.union("º", "ª")
        # set of symbols over which the rule operates
        _sigma_star = pynini.union(_digit, _powers).closure().optimize()

        # Inserts factors in the appropriate place in the digit sequence.
        _raw_factorizer = (
            _digit + pynutil.insert("[E9]") + _digit + pynutil.insert("[E2]") + _digit +
            pynutil.insert("[E1]") + _digit + pynutil.insert("[E6]") + _digit +
            pynutil.insert("[E2]") + _digit + pynutil.insert("[E1]") + _digit +
            pynutil.insert("[E3]") + _digit + pynutil.insert("[E2]") + _digit +
            pynutil.insert("[E1]") + _digit)

        # Deletes "0" and "0" followed by a factor, so as to clear out unverbalized
        # material in cases like "2,000,324". This needs to be done with some care since
        # we need to keep E3 if it has a multiplier, but not if there is nothing between
        # the thousands and the millions place.
        _del_zeros = (
            # [end of string]->regex $ (deletes trailing zeros)
            pynini.cdrewrite(pynutil.delete("0"), "", "[EOS]", _sigma_star)
            @ pynini.cdrewrite(pynutil.delete("0[E1]"), "", "", _sigma_star)
            @ pynini.cdrewrite(pynutil.delete("0[E2]"), "", "", _sigma_star)
            @ pynini.cdrewrite(pynutil.delete("0[E3]"), "[E6]", "", _sigma_star)
            @ pynini.cdrewrite(pynutil.delete("0[E6]"), "[E9]", "", _sigma_star)
            @ pynini.cdrewrite(pynutil.delete("0[E9]"), "", "", _sigma_star)
            @ pynini.cdrewrite(pynutil.delete("0"), "", "", _sigma_star)
        ).optimize()
        # Inserts an arbitrary number of zeros at the beginning of a string so that
        # shorter strings can match the length expected by the raw factorizer.
        _pad_zeros = pynutil.insert(
            "0").closure().concat(pynini.closure(_digit))

        # Changes E1 to E1* for 11-19.
        _fix_teens = pynini.cdrewrite(
            pynini.cross("[E1]", "[E1*]"), "1", _digit, _sigma_star)

        # The actual factorizer
        _phi = (_pad_zeros @ _raw_factorizer @
                _del_zeros @ _fix_teens).optimize()

        masc = pynini.cross("º", "o")
        fem = pynini.cross("ª", "a")

        digits_sing = pynini.string_map([("unesim", "prim"), ("duesim", "second"), ("tresim", "terz"),
                                         ("quattresim", "quart"), ("cinquesim",
                                                                   "quint"), ("seesim", "sest"),
                                         ("settesim", "settim"), ("ottesim",
                                                                  "ottav"), ("novesim", "non"),
                                         ("diecesim", "decim")])

        base = pynini.string_map([
            ("1", "uno"), ("2", "due"), ("3", "tre"),
            ("4", "quattro"), ("5",
                               "cinque"), ("6", "sei"),
            ("7", "sette"), ("8", "otto"), ("9", "nove"),
            ("1[E1]", "dieci"), ("1[E1*]1", "undici"),
            ("1[E1*]2", "dodici"), ("1[E1*]3", "tredici"),
            ("1[E1*]4", "quattordici"), ("1[E1*]5",
                                         "quindici"),
            ("1[E1*]6", "sedici"), ("1[E1*]7",
                                    "diciassette"),
            ("1[E1*]8", "diciotto"), ("1[E1*]9",
                                      "diciannove"),
            ("2[E1]", "venti"), ("3[E1]", "trenta"),
            ("4[E1]", "quaranta"), ("5[E1]", "cinquanta"),
            ("6[E1]", "sessanta"), ("7[E1]", "settanta"),
            ("8[E1]", "ottanta"), ("9[E1]", "novanta")]).optimize()
        powers = pynini.string_map([
            ("[E2]", "cento"),
            ("[E3]", "mila"),
            ("[E6]", "milioni"),
            ("[E9]", "miliardi")
        ])

        cut_end = pynini.string_map([
            ("venti", "vent"),
            ("trenta", "trent"),
            ("quaranta", "quarant"),
            ("cinquanta", "cinquant"),
            ("sessanta", "sessant"),
            ("settanta", "settant"),
            ("ottanta", "ottant"),
            ("novanta", "novant"),
        ])

        plural_to_singular = pynini.string_map([
            ("mila", "mille"),
            ("milioni", "milione"),
            ("miliardi", "miliardo")
        ])

        zero = pynini.cross("0", "zero")

        letters = pynini.union(*string.ascii_lowercase)
        sigma_star = pynini.union(letters, ' ').closure().optimize()

        _exceptions = (
            pynini.cdrewrite(cut_end, "", pynini.union(
                " uno", " otto"), sigma_star)
            @ pynini.cdrewrite(plural_to_singular, "uno ", "", sigma_star)
            @ pynini.cdrewrite(pynutil.delete("uno "), "", pynini.union("mille", "cento"), sigma_star)
            @ pynini.cdrewrite(pynini.cross("uno ", "un "), "", pynini.union("milione", "miliardo"), sigma_star)
            @ pynini.cdrewrite(pynini.cross("mille", "uno mila"), "cento ", "", sigma_star)
            @ pynini.cdrewrite(pynini.cross("un milione", "uno milioni"), "cento ", "", sigma_star)
        ).optimize()

        sigma_star = pynini.union(sigma_star, ordinal).closure().optimize()

        _ordinals = (
            pynini.cdrewrite(pynutil.delete(pynini.union(
                "a", "e", "i", "o")), "", "[EOS]", sigma_star)
            @ pynini.cdrewrite(pynutil.insert("esim"), "", "[EOS]", sigma_star)
            @ pynini.cdrewrite(digits_sing, "[BOS]", "[EOS]", sigma_star)
        ).optimize()

        _lambda = base | powers

        _lambda_star = pynutil.join(_lambda, pynutil.insert(" ")).optimize()
        graph = (_phi @ _lambda_star @ _exceptions @ _ordinals) + (masc | fem)
        graph.optimize()

        final_graph = pynutil.insert(
            "integer: \"") + graph + pynutil.insert("\"")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
