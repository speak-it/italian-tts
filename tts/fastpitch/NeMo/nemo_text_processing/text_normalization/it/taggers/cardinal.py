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

cardinal_separator = pynini.string_map([".", NEMO_SPACE])


def filter_punctuation(fst: 'pynini.FstLike') -> 'pynini.FstLike':
    """
    Helper function for parsing number strings. Converts common cardinal strings (groups of three digits delineated by 'cardinal_separator' - see graph_utils)
    and converts to a string of digits:
        "1 000" -> "1000"
        "1.000.000" -> "1000000"
    Args:
        fst: Any pynini.FstLike object. Function composes fst onto string parser fst

    Returns:
        fst: A pynini.FstLike object
    """
    exactly_three_digits = NEMO_DIGIT ** 3  # for blocks of three
    up_to_three_digits = pynini.closure(
        NEMO_DIGIT, 1, 3)  # for start of string

    cardinal_string = pynini.closure(
        NEMO_DIGIT, 1
    )  # For string w/o punctuation (used for page numbers, thousand series)

    cardinal_string |= (
        up_to_three_digits
        + pynutil.delete(cardinal_separator)
        + pynini.closure(exactly_three_digits +
                         pynutil.delete(cardinal_separator))
        + exactly_three_digits
    )

    return cardinal_string @ fst


class CardinalFst(GraphFst):
    def __init__(self):
        super().__init__(name="cardinal", kind="classify")

        _digit = pynini.union(*string.digits)  # 0 | 1 | ... | 9
        # Powers of ten that have single-word representations in English. E1*
        # is a special symbol we will use in the teens below.
        _powers = pynini.union("[E1]", "[E1*]", "[E2]", "[E3]", "[E6]", "[E9]")
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

        base = pynini.string_map([
            ("1", "uno"), ("2", "due"), ("3", "tre"),
            ("4", "quattro"), ("5", "cinque"), ("6", "sei"),
            ("7", "sette"), ("8", "otto"), ("9", "nove"),
            ("1[E1]", "dieci"), ("1[E1*]1", "undici"),
            ("1[E1*]2", "dodici"), ("1[E1*]3", "tredici"),
            ("1[E1*]4", "quattordici"), ("1[E1*]5", "quindici"),
            ("1[E1*]6", "sedici"), ("1[E1*]7", "diciassette"),
            ("1[E1*]8", "diciotto"), ("1[E1*]9", "diciannove"),
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

        _lambda = base | powers

        _lambda_star = pynutil.join(_lambda, pynutil.insert(" ")).optimize()

        cifre = pynini.string_map([
            ("0", "zero"), ("1", "uno"), ("2", "due"), ("3", "tre"),
            ("4", "quattro"), ("5", "cinque"), ("6", "sei"),
            ("7", "sette"), ("8", "otto"), ("9", "nove")
        ])
        self.cifre = pynutil.add_weight(pynutil.join(
            cifre, pynutil.insert(" ")), .1).optimize()

        graph = (_phi @ _lambda_star @ _exceptions | zero | cifre)
        self.graph = filter_punctuation(graph).optimize()

        # arriva fino a 10 miliardi (escluso)
        # se più lungo lo legge cifra per cifra

        negatives = pynini.closure(pynutil.insert(
            "negative: ") + pynini.cross("-", "\"true\" "), 0, 1)
        positives = pynini.closure(pynutil.insert(
            "positive: ") + pynini.cross("+", "\"true\" "), 0, 1)

        final_graph = positives + negatives + \
            pynutil.insert("integer: \"") + self.graph + pynutil.insert("\"")
        final_graph = self.add_tokens(final_graph)
        self.fst = final_graph.optimize()
