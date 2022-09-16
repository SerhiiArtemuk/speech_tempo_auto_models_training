import os
import tqdm
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
import pickle as pkl
from sklearn.metrics import mean_absolute_percentage_error, r2_score

from tools.convert_to_ipa import convert_to_ipa
from tools.get_audio_length import get_audio_length
from tools.transliterate import transliterate, translate
from google_tts import google_tts_syntesize


_DATASET_PATH = Path("/home/pc/Dev/train-clean-100/dataset_v2.csv")
_N_SAMPLES = 600 
_OUT_OVERWRITE = False
_CHARACTERS_COUNTER = 0
_FREE_TTS_CHARACTERS = 1_000_000

IGNORE_CHARS = "?.¿!-?;:,"

LANG_DICT = pd.read_csv("vidby.csv")
LANG_DICT["transliterate"] = LANG_DICT["transliterate"].isna().apply(lambda x : not(x))
LANG_DICT = LANG_DICT.dropna()


def tts(text, speaker, out_file=None):

    google_tts_syntesize(text=text, voice=speaker)

    if os.path.exists("./tmp.wav"):
        # move file to needed place if needed
        if out_file is None:
            out_file = "./tmp.wav"
        else:
            shutil.move("./tmp.wav", out_file)
        return True
    else:
        return False

def prepare_text(text):
    for char in IGNORE_CHARS:
        text = text.replace(char, "")
    text = text.lower()
    return text

def _plot_data(lang_df: pd.DataFrame) -> bool:
    
    """
    This function plot evaluated data to assess the dependence of the predicted 
    duration of the phrase on the ground truth.

    Args:
        lang_df (pd.DataFrame): dataframe with all necessery information to make
        plot. 

    Returns:
        bool: True if the _plot_data() ended successfully and False if not. 
    """
    
    # Creating folder to save plot results (graph and data)
    os.makedirs("plots", exist_ok=True)
    
    language = lang_df['lang_name'][0]
    speaker = lang_df['lang_speaker'][0]
    lang_code = lang_df['lang_code'][0]

    fig, axs = plt.subplots(3, 1, figsize=(6, 9))
    fig.suptitle(f"{language}. Duration prediction.")

    # Get data
    n = lang_df["n_chars"]
    t_true = lang_df["t_true"]
    t_pred = lang_df["t_pred"]

    # Plot dependencies
    axs[0].set_title(f"Speaker: {speaker}\nDuration prediction.")
    axs[0].scatter(n, t_true, label=f"t_true")
    axs[0].scatter(n, t_pred, label=f"t_pred")
    axs[0].set_ylabel("time, seconds")
    
    # Plot difference
    t_diff = lang_df["t_diff"]
    axs[1].set_title(f"Difference (t_true - t_pred).")
    axs[1].scatter(n, t_diff, label=f"t_diff")
    axs[1].set_ylabel("time difference, seconds")

    # Plor error
    t_ratio = lang_df["t_ratio"]
    t_ratio = t_ratio.apply(lambda x : abs(x) * 100.0)
    axs[2].set_title(f"Error (t_true - t_pred) / t_true).")
    axs[2].scatter(n, t_ratio, label=f"t_ratio")
    axs[2].set_ylabel("error, %")
    axs[2].set_xlabel("number of phonemes")

    for ax in axs.flat:
        ax.label_outer()
        ax.grid()
        ax.legend()
        
    plt.savefig(f"plots/summary_evaluate_{lang_code}.png")
    plt.cla()
    plt.clf()

    return True

def _eval_data(model, lang_name, lang_code, lang_speaker, do_translit, lang_ipa, lang_translate):

    # Stage 3: Evaluate  
    print("Evaluate...")
    eval_list = []              
    lang_text_file = f"dataset/test-set/{lang_code}.txt"

    print(f"Processing {lang_name} ({lang_code})")

    # Create folder to save evaluation data
    speaker_dir_eval = f"data/evaluate_backup/audio/{lang_code}/{lang_speaker}"
    
    
    os.makedirs(speaker_dir_eval, exist_ok=True)

    with open(lang_text_file, "r") as f:
        phrases = f.readlines()

    for i, text in tqdm.tqdm(enumerate(phrases), total=len(phrases)):
        
        text = prepare_text(text)
        out_file = os.path.join(speaker_dir_eval, f"{str(i).zfill(4)}.wav")
        
        if os.path.exists(out_file):
            success = True
        else:
            success = tts(text, lang_speaker, out_file=out_file)

        if success:
            global _CHARACTERS_COUNTER
            _CHARACTERS_COUNTER += len(text)
            if do_translit:
                transliteration = transliterate(text, lang_translate)
                text_ipa = convert_to_ipa(transliteration, lang="en")
            else:
                text_ipa = convert_to_ipa(text, lang=lang_ipa)

            n_chars = len(text_ipa)

            t_true = get_audio_length(out_file)
            t_pred = model.predict([[n_chars]])[0]
            t_diff = t_true - t_pred
            t_ratio = t_diff / t_true

            eval_list.append(
                (lang_name, lang_code, lang_speaker, t_true, t_pred, t_diff, t_ratio, n_chars, text, text_ipa)
            )

    eval_df = pd.DataFrame(
        data=eval_list,
        columns=("lang_name", "lang_code", "lang_speaker", "t_true", "t_pred", "t_diff", "t_ratio", "n_chars", "text", "text_ipa"),
    )

    ev_data_save_path = f"data/evaluate_backup/gen_data_eval"
    os.makedirs(ev_data_save_path, exist_ok=True)
    eval_df.to_csv(ev_data_save_path + f'/out_{lang_code}.csv', index=False)
    
    summary = []
    error = eval_df["t_ratio"].apply(lambda x : abs(x) * 100.0)
    mape = mean_absolute_percentage_error(eval_df['t_true'].values, 
                                            eval_df['t_pred'].values)
    r2 = r2_score(eval_df['t_true'].values, eval_df['t_pred'].values)

    # TODO remake summary logic
    summary.append((
        lang_name,
        error.mean(),
        error.max(),
        error.min(),
        error.std(),
        mape,
        r2
    ))

    summary_df = pd.DataFrame(
    data=summary,
    columns=(
        "language",
        "error_mean",
        "error_max",
        "error_min",
        "error_std",
        "MAPE",
        "R2"
    ),
    )
    os.makedirs('data/evaluate_backup/error_csv', exist_ok=True)
    summary_df.to_csv(f"data/evaluate_backup/error_csv/out_summary_{lang_code}.csv")

    return eval_df

def _train_data(out_df, lang_code):

    print("Train model...")

    X = out_df["n_symbols"].to_numpy()
    X = np.reshape(X, (-1, 1))
    y = out_df["seconds"].to_numpy()

    model = LinearRegression().fit(X, y)

    model_path = f'models/linear_regression_{lang_code}'
    os.makedirs(model_path, exist_ok=True)

    with open(model_path + '/model.pkl', 'wb') as f:
        pkl.dump(model, f)

    return model

def _prepare_data(speaker_dir, train_data_file, process, lang_translate, lang_speaker, do_translit, lang_ipa):

    os.makedirs(speaker_dir, exist_ok=True)
    out_data = []

    with open(train_data_file, "r") as f:
        train_phrases = f.readlines()

    for i, text in tqdm.tqdm(enumerate(train_phrases), total=len(train_phrases)):
        
        text = prepare_text(text)
        wav_path = os.path.join(speaker_dir, f"{str(i).zfill(4)}.wav")

        if os.path.exists(wav_path):
            success = True
        else:
            if process != "train":
                translated_text = translate(text, lang_translate, 'en')
                translated_text = prepare_text(translated_text)
                text = translated_text
            success = tts(text, lang_speaker, out_file=wav_path)

        if success:
            global _CHARACTERS_COUNTER
            _CHARACTERS_COUNTER += len(text)

            if do_translit:
                # Transliterate text
                transliteration = transliterate(text, lang_translate)
                # Convert to IPA
                ipa_text = convert_to_ipa(transliteration, lang="en")
            else:
                # Convert to IPA
                ipa_text = convert_to_ipa(text, lang=lang_ipa)

            seconds = get_audio_length(wav_path)

            # Save data into list
            out_data.append((seconds, len(ipa_text), ipa_text, text))
    return out_data

def _prepare_train_data(lang_code, lang_speaker, lang_translate, do_translit,
                        lang_ipa):

    

    # Create folder for trained data
    out_train_data_path = f'data/train_data_backup/gen_data_csv/{lang_code}'
    os.makedirs(out_train_data_path, exist_ok=True)
    train_data_filename = out_train_data_path + f"/gen_data_{lang_code}.csv"
    
    # Stage 1: Prepare Data 

    print("Prepare data...")

    # Read dataset
    df = pd.read_csv(_DATASET_PATH)

    out_data = []
    # TODO lang_code insert
    train_data_file = f'dataset/train/set/en-US.txt'

    speaker_dir = f"data/train_data_backup/audio/{lang_code}/{lang_speaker}"
    os.makedirs(speaker_dir, exist_ok=True)

    with open(train_data_file, "r") as f:
        train_phrases = f.readlines()

    for i, text in tqdm.tqdm(enumerate(train_phrases), total=len(train_phrases)):
        
        text = prepare_text(text)
        wav_path = os.path.join(speaker_dir, f"{str(i).zfill(4)}.wav")

        if os.path.exists(wav_path):
            success = True
        else:
            translated_text = translate(text, lang_translate, 'en')
            translated_text = prepare_text(translated_text)
            success = tts(translated_text, lang_speaker, out_file=wav_path)

        if success:
            global _CHARACTERS_COUNTER
            _CHARACTERS_COUNTER += len(translated_text)

            seconds = get_audio_length(wav_path)

            if do_translit:
                # Transliterate text
                transliteration = transliterate(translated_text, lang_translate)
                # Convert to IPA
                ipa_text = convert_to_ipa(transliteration, lang="en")
            else:
                # Convert to IPA
                ipa_text = convert_to_ipa(translated_text, lang=lang_ipa)

            # Save data into list
            out_data.append((seconds, len(ipa_text), ipa_text, text, translated_text))

        
    # Create df and sace data
    out_df = pd.DataFrame(out_data, columns=("seconds", "n_symbols", "transcription", "text", "translated_texts"))
    out_df.to_csv(train_data_filename)

    return out_df

def main():

    global _CHARACTERS_COUNTER

    done_models = ['zh-CN', 'fr-FR', 'ja-JP', 'af-ZA', 'ca-ES', 'da-DK', 
    'de-DE', 'fr-CA', 'gu-IN', 'hi-IN', 'id-ID', 'ko-KR', 'lv-LV', 'ml-IN', 
    'pt-BR', 'pt-PT', 'sr-RS', 'sv-SE', 'ta-IN', 'te-IN', 'th-TH', 'vi-VN',
    'yue-HK']

    lang_to_make_model = ['zh-CN']

    for idx, row in LANG_DICT.iterrows():

        lang_ipa = row["language_ipa"]
        lang_name = row["language_name"]
        lang_code = row["language_code"]
        lang_speaker = row["google_speaker"]
        do_translit= row["transliterate"]
        lang_translate = row["language_translate"]
        lang_params_dict = row.to_dict()

        # Check if lang in models list
        if lang_code not in lang_to_make_model:
            print(f'Lang code {lang_code} not in models list. Continue ...')
            continue

        print(f'Processed: {lang_name}, {lang_code}, {lang_speaker}')
        print(f'Need transliteration? -> {do_translit}')

        
        out_df = _prepare_train_data(lang_code=lang_code,
                                    lang_speaker=lang_speaker,
                                    lang_translate=lang_translate,
                                    do_translit=do_translit,
                                    lang_ipa=lang_ipa
        )

        # Stage 2: Train
        model = _train_data(out_df, lang_code=lang_code)

        # Stage 3: Evaluate
        eval_df = _eval_data(model=model, lang_name=lang_name, 
                            lang_code=lang_code, lang_speaker=lang_speaker,
                            do_translit=do_translit, lang_ipa=lang_ipa, 
                            lang_translate=lang_translate)

        # Stage 4: Plot Results
        _plot_data(eval_df)
        
        # Checking whether the number of used symbols does not exceed 
        # the number of free TTS symbols 
        print(f'Train, Evaluate, Plot for {lang_name} -> done!')
        print(f'Spend characters for TTS: {_CHARACTERS_COUNTER}')
        if int(_CHARACTERS_COUNTER) >= (_FREE_TTS_CHARACTERS - 200_000):
            print('Ran out of free symbols for TTS. Continue? y/N')
            decision = input()
            if decision.lower() == 'y':
                continue
            else:
                break
        print('-------------------------------------------------------------')

    print("Done!")


if __name__ == "__main__":
    main()