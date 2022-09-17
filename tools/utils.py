from googletrans import Translator
import soundfile as sf
import subprocess
from shlex import quote

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


def get_audio_length(path=None, audio=None, sr=None):
    if path is not None:
        audio, sr = sf.read(path)
    elif audio is None or sr is None:
        raise ValueError("`audio` and `sr` or `path` must not be None!")
    return len(audio) / sr


def convert_to_ipa(text, lang):
    output = subprocess.run(
        ["espeak", "-v", lang, "--ipa", quote(text), "-w", "/dev/null"], 
        capture_output=True,
    )
    output = output.stdout.decode("utf-8").strip()
    return output
    
