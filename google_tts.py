import requests as re
from requests.structures import CaseInsensitiveDict
import base64
import subprocess
import os

# _API_KEY = "AIzaSyAKVOK650Yv8zPbSWK_6AOtGWBmVhHsYRE" # Vitalii
# _API_KEY = "AIzaSyDCpAWVfuYexA-zfq6cQZ3ijkb3uVLnkio" # Serhii
_API_KEY = "AIzaSyB6CVDnSSBvmxWn046Ol3ZkJKMenmf_5u0" # Unidatalab Serhii
_API_NAME = "X-Goog-Api-Key"

def google_tts_syntesize(text, voice, lang_code='en-US'):
    
    url = "https://texttospeech.googleapis.com/v1beta1/text:synthesize"
    headers = CaseInsensitiveDict()
    
    headers["X-Goog-Api-Key"] = _API_KEY
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