import os

class Config:

    ROOT = os.path.dirname(os.path.realpath(__file__))

    TTS_API_KEY = '' # Place TTS API KEY
    TTS_API_NAME = 'X-Goog-Api-Key'

    DEFAULT_MODEL_LANG_CODE = 'en-US'
    DEFAULT_MODEL_PATH = os.path.join(ROOT,
                f"models/linear_regression_{DEFAULT_MODEL_LANG_CODE}/model.pkl")

    # Place here language code which used baseline model to predict speech duration
    BASELINE_LANGS = ['ar-XA', 'bn-IN', 'bg-BG', 'cs-CZ', 'nl-BE', 'nl-NL',
                    'en-IN', 'en-GB', 'fi-FI', 'el-GR', 'hu-HU', 'en-US',
                    'is-IS', 'it-IT', 'kn-IN', 'ms-MY', 'pl-PL', 'pa-IN',
                    'ro-RO', 'ru-RU', 'sk-SK', 'es-ES', 'es-US', 'tr-TR',
                    'uk-UA']

    # Place here language code wich needed custom models training
    LANGUAGE_TO_TRAIN = ['fr-FR', 'ja-JP', 'af-ZA', 'ca-ES', 'da-DK', 
    'de-DE', 'fr-CA', 'hi-IN', 'id-ID', 'ko-KR', 'lv-LV', 'ml-IN', 'tr-TR', 
    'pt-BR', 'pt-PT', 'sr-RS', 'sv-SE', 'ta-IN', 'te-IN', 'vi-VN', 'uk-UA',
    'ar-XA', 'bn-IN', 'bg-BG', 'cs-CZ', 'nl-BE', 'nl-NL', 'en-IN', 'es-US',
    'en-GB', 'fi-FI', 'el-GR', 'hu-HU', 'en-US', 'is-IS', 'it-IT', 'kn-IN',
    'ms-MY', 'pl-PL', 'pa-IN', 'ro-RO', 'ru-RU', 'sk-SK', 'es-ES']