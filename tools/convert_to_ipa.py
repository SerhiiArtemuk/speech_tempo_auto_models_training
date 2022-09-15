import subprocess
from shlex import quote

def convert_to_ipa(text, lang):
    output = subprocess.run(
        ["espeak", "-v", lang, "--ipa", quote(text), "-w", "/dev/null"], 
        capture_output=True,
    )
    output = output.stdout.decode("utf-8").strip()
    return output
    
