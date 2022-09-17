import base64
import subprocess
import os
import requests as re
from config import Config
from requests.structures import CaseInsensitiveDict

_API_KEY = Config.TTS_API_KEY
_API_NAME = Config.TTS_API_NAME

def google_tts_syntesize(text, voice, lang_code='en-US'):
    
    url = "https://texttospeech.googleapis.com/v1beta1/text:synthesize"
    headers = CaseInsensitiveDict()
    
    headers[_API_NAME] = _API_KEY
    headers["Content-Type"] = "application/json; charset=UTF-8"
    body = {
        "input":{
            "text": text,
        },
        "voice":{
            "languageCode": lang_code,
            "name": voice,
        },
        "audioConfig":{
            "audioEncoding":"mp3",
        }
    }

    res = re.post(url, json = body, headers = headers, auth = (_API_NAME, _API_KEY))

    data = res.json()
    try:
        audio_content = data['audioContent']

        wav_data = base64.b64decode(audio_content)

        with open('tmp.mp3', 'bx') as f:
            
            f.write(wav_data)

        subprocess.call([
            "ffmpeg", 
            "-i", "tmp.mp3", "tmp.wav",
            "-hide_banner",
            "-loglevel", "error", "-y"
        ])

        os.remove("tmp.mp3")
    except:
        print('Audio content does`nt exist')

if __name__ == "__main__":
    google_tts_syntesize("et en présence de tout le peuple tout ce qu'il avait dit étant demandé par holopherne", 
    'fr-FR-Wavenet-A')