from .NeMo.nemo.collections.tts.torch.tts_tokenizers import BaseCharsTokenizer


def italian_text_preprocessing(text):
    return text.lower()


class ItalianCharsTokenizer(BaseCharsTokenizer):
    def __init__(
        self,
        punct=True,
        apostrophe=True,
        add_blank_at=None,
        pad_with_space=False,
        non_default_punct_list=None,
        text_preprocessing_func=italian_text_preprocessing,
        phonemes=True
    ):
        it_alphabet = "abcdefghijklmnopqrstuvwxyzàèéìîíòóùú"
        if phonemes:
            #it_ipa = "d͡zd͡ʒʎɱŋɲʃt͡st͡ʃθʒɛɔøɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
            it_ipa = "d͡zd͡ʒʎɱŋɲʃt͡st͡ʃθʒɛɔøˈˌ.ː"
            it_alphabet += it_ipa
        super().__init__(
            chars=it_alphabet,
            punct=punct,
            apostrophe=apostrophe,
            add_blank_at=add_blank_at,
            pad_with_space=pad_with_space,
            non_default_punct_list=non_default_punct_list,
            text_preprocessing_func=text_preprocessing_func
        )
