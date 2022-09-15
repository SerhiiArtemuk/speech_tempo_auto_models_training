from googletrans import Translator


translator = Translator()


def transliterate(text: str, src_lang: str = None):
    if src_lang is None:
        detection = translator.detect(text)
        src_lang = detection.lang
        if isinstance(src_lang, list):
            confidence = detection.confidence
            src_lang = src_lang[confidence.index(max(confidence))]
    return translator.translate(text, dest=src_lang, src=src_lang).pronunciation


def translate(text: str, dst_lang: str, src_lang: str = "auto"):
    return translator.translate(text, dest=dst_lang, src=src_lang).text
