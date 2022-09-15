import soundfile as sf


def get_audio_length(path=None, audio=None, sr=None):
    if path is not None:
        audio, sr = sf.read(path)
    elif audio is None or sr is None:
        raise ValueError("`audio` and `sr` or `path` must not be None!")
    return len(audio) / sr

