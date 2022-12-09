from fastpitch.NeMo.nemo_text_processing.text_normalization.en.graph_utils import GraphFst
from fastpitch.NeMo.nemo_text_processing.text_normalization.it.verbalizers.whitelist import WhiteListFst
from fastpitch.NeMo.nemo_text_processing.text_normalization.it.verbalizers.cardinal import CardinalFst
from fastpitch.NeMo.nemo_text_processing.text_normalization.it.verbalizers.decimal import DecimalFst
from fastpitch.NeMo.nemo_text_processing.text_normalization.it.verbalizers.ordinal import OrdinalFst
from fastpitch.NeMo.nemo_text_processing.text_normalization.it.verbalizers.money import MoneyFst
from fastpitch.NeMo.nemo_text_processing.text_normalization.it.verbalizers.measure import MeasureFst
from fastpitch.NeMo.nemo_text_processing.text_normalization.it.verbalizers.time import TimeFst
from fastpitch.NeMo.nemo_text_processing.text_normalization.it.verbalizers.electronic import ElectronicFst
from fastpitch.NeMo.nemo_text_processing.text_normalization.it.verbalizers.abbreviation import AbbreviationFst


class VerbalizeFst(GraphFst):
    """
    Composes other verbalizer grammars.
    For deployment, this grammar will be compiled and exported to OpenFst Finate State Archiv (FAR) File.
    More details to deployment at NeMo/tools/text_processing_deployment.

    Args:
        deterministic: if True will provide a single transduction option,
            for False multiple options (used for audio-based normalization)
    """

    def __init__(self, deterministic: bool = True):
        super().__init__(name="verbalize", kind="verbalize", deterministic=deterministic)
        cardinal = CardinalFst()
        cardinal_graph = cardinal.fst
        ordinal = OrdinalFst()
        ordinal_graph = ordinal.fst
        decimal = DecimalFst()
        decimal_graph = decimal.fst
        money = MoneyFst(decimal)
        money_graph = money.fst
        measure = MeasureFst(decimal, cardinal)
        measure_graph = measure.fst
        whitelist_graph = WhiteListFst().fst
        time_graph = TimeFst().fst
        electronic_graph = ElectronicFst().fst
        abbreviation_graph = AbbreviationFst().fst
        graph = (
            cardinal_graph
            | decimal_graph
            | ordinal_graph
            | whitelist_graph
            | money_graph
            | measure_graph
            | time_graph
            | electronic_graph
            | abbreviation_graph
        )
        self.fst = graph
