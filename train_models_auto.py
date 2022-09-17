import os
import tqdm
import shutil
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from config import Config
from tools.utils import convert_to_ipa, get_audio_length, transliterate
from tools.google_tts import google_tts_syntesize
from logger_tools import setup_logger

logger = setup_logger('train_models_auto')

_CHARACTERS_COUNTER = 0
_FREE_TTS_CHARACTERS = 1_000_000
IGNORE_CHARS = "?.¿!-?;:,"
ROOT = Config.ROOT
LANG_DICT = pd.read_csv(os.path.join(ROOT, 'look_up.csv'))
LANG_DICT["transliterate"] = LANG_DICT["transliterate"].isna().apply(lambda x : not(x))
LANG_DICT = LANG_DICT.dropna()


def tts(text: str, speaker:  str, out_file: str = None) -> bool:

    """
    This function generate wav audio file using google TTS engine

    Args:
        text (str): text to synthesize
        speaker (str): speaker code for google TTS engine

    Returns:
        bool: True if TTS finished successsfuly
    """

    google_tts_syntesize(text=text, voice=speaker)

    if os.path.exists("./tmp.wav"):
        
        if out_file is None:
            out_file = "./tmp.wav"
        else:
            shutil.move("./tmp.wav", out_file)
        return True
    else:
        return False

def prepare_text(text: str) -> str:

    """
    This function remove IGNORE_CHARS from text

    Args:
        text (str): text to clean up

    Returns:
        str: cleaned text
    """

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

    logger.info(f'Plots saved!')
    return True

def _eval_data(model, lang_name: str, lang_code: str, lang_speaker: str, 
                do_translit: bool, lang_ipa: str, lang_translate: str) -> pd.DataFrame:
    """
    This function evaluates trained data using test-set

    Args:
        model: model to make predict
        lang_name (str): language name
        lang_code (str): language code
        lang_speaker (str): speaker code
        do_translit (bool): True or False
        lang_ipa (str): language code for ipa (eSpeak)
        lang_translate (str): language code for transliteration

    Returns:
        pd.DataFrame: dataframe with evaluation data
    """

    process = 'evaluate'
    eval_list = []              
    lang_text_file = f"dataset/test-set/{lang_code}.txt"
    speaker_dir_eval = f"data/evaluate_backup/audio/{lang_code}/{lang_speaker}"
    out_df = _prepare_data(process, speaker_dir_eval, lang_text_file, lang_translate, 
                                lang_speaker, do_translit, lang_ipa)

    logger.info(f"Evaluate model for {lang_code}...")

    for i, row in out_df.iterrows():

        n_chars = row['n_symbols']
        t_true = get_audio_length(row['wav_path'])
        t_pred = model.predict([[n_chars]])[0]
        t_diff = t_true - t_pred
        t_ratio = t_diff / t_true
        eval_list.append(
            (lang_name, lang_code, lang_speaker, t_true, t_pred, t_diff, t_ratio, n_chars, row['text'], row['transcription'])
        )

    eval_df = pd.DataFrame(
        data=eval_list,
        columns=("lang_name", "lang_code", "lang_speaker", "t_true", "t_pred", "t_diff", "t_ratio", "n_chars", "text", "text_ipa"),
    )

    logger.info(f'Saving evaluating results ...')
    ev_data_save_path = f"data/evaluate_backup/gen_data_eval"
    os.makedirs(ev_data_save_path, exist_ok=True)
    eval_df.to_csv(ev_data_save_path + f'/out_{lang_code}.csv', index=False)
    
    error = eval_df["t_ratio"].apply(lambda x : abs(x) * 100.0)
    r2 = r2_score(eval_df['t_true'].values, eval_df['t_pred'].values) * 100

    summary = (
        lang_name,
        error.mean(),
        error.max(),
        error.min(),
        error.std(),
        r2
    )

    summary_df = pd.DataFrame(
    data=[summary],
    columns=(
        "language",
        "error_mean",
        "error_max",
        "error_min",
        "error_std",
        "R2_score"
    ),
    )
    os.makedirs('data/evaluate_backup/error_csv', exist_ok=True)
    summary_df.to_csv(f"data/evaluate_backup/error_csv/out_summary_{lang_code}.csv",
                        index=False)
    logger.info('Evaluation results saved!')
    return eval_df

def _train_data(lang_code: str,  lang_translate: str, lang_speaker: str,
                do_translit: str, lang_ipa: str) -> LinearRegression():
    """
    This function train LinearRegression model using train set

    Args:
        lang_code (str): language code
        lang_translate (str): language code for tranliteration part
        lang_speaker (str): language speaker code
        do_translit (str): True or False
        lang_ipa (str): language code for ipa (eSpeak)

    Returns:
        model (LinearRegression): Linear regression model
    """
    process = 'train'
    train_set_text_file = f'dataset/train-set/{lang_code}.txt'
    speaker_dir = f"data/train_data_backup/audio/{lang_code}/{lang_speaker}"
    out_df = _prepare_data(process, speaker_dir, train_set_text_file, lang_translate, 
                                lang_speaker, do_translit, lang_ipa)
    out_train_data_path = f'data/train_data_backup/gen_data_csv/{lang_code}'
    os.makedirs(out_train_data_path, exist_ok=True)
    train_data_filename = out_train_data_path + f"/gen_data_{lang_code}.csv"
    out_df.to_csv(train_data_filename, index=False)

    logger.info(f"Train model for {lang_code}")
    X = out_df["n_symbols"].to_numpy()
    X = np.reshape(X, (-1, 1))
    y = out_df["seconds"].to_numpy()
    model = LinearRegression().fit(X, y)

    logger.info('Training done. Saving model ...')
    model_path = f'models/linear_regression_{lang_code}'
    os.makedirs(model_path, exist_ok=True)

    with open(model_path + '/model.pkl', 'wb') as f:
        pkl.dump(model, f)
    logger.info(f'Model saved!')

    return model

def _prepare_data(process: str, audio_dir: str, data_file: str,
                lang_translate: str, lang_speaker: str, do_translit: bool,
                lang_ipa: str) -> pd.DataFrame:

    """
    This function prepare data for train and evaluate

    Args:
        process (str): current process
        audio_dir (str): directory to save audio
        data_file (str): file w train or test data
        lang_translate (str): language code to transliteration
        lang_speaker (str): speaker code for TTS
        do_translit (bool): True or False
        lang_ipa (str): language code for IPA 

    Returns:
        pd.DataFrame: dataframe with prepared data
    """

    logger.info(f'Start data preparing for {process} process.')
    os.makedirs(audio_dir, exist_ok=True)
    out_data = []

    with open(data_file, "r") as f:
        train_phrases = f.readlines()

    for i, text in tqdm.tqdm(enumerate(train_phrases), total=len(train_phrases)):
        text = prepare_text(text)
        wav_path = os.path.join(audio_dir, f"{str(i).zfill(4)}.wav")

        if os.path.exists(wav_path):
            success = True
        else:
            success = tts(text, lang_speaker, out_file=wav_path)
            global _CHARACTERS_COUNTER
            _CHARACTERS_COUNTER += len(text)

        if success:
            if do_translit:
                transliteration = transliterate(text, lang_translate)
                ipa_text = convert_to_ipa(transliteration, lang="en")
            else:
                ipa_text = convert_to_ipa(text, lang=lang_ipa)

            seconds = get_audio_length(wav_path)
            out_data.append((seconds, len(ipa_text), ipa_text, text, wav_path))
    out_df = pd.DataFrame(out_data, columns=("seconds", "n_symbols", "transcription", "text", "wav_path"))
    return out_df

def main():

    global _CHARACTERS_COUNTER

    logger.info(f'Start automatic train process for languages: {Config.LANGUAGE_TO_TRAIN}')

    for l in Config.LANGUAGE_TO_TRAIN:

        logger.info(f'Current processed lang: {l}')
        if l not in LANG_DICT['language_code'].tolist():
            logger.info(f'Lang code {lang_code} not in look up table. Skipping ...')
            continue
        
        processed_row = LANG_DICT[LANG_DICT['language_code'] == l]
        lang_ipa = processed_row["language_ipa"].values[0]
        lang_name = processed_row["language_name"].values[0]
        lang_code = processed_row["language_code"].values[0]
        lang_speaker = processed_row["google_speaker"].values[0]
        do_translit= processed_row["transliterate"].values[0]
        lang_translate = processed_row["language_translate"].values[0]

        logger.info('Start training!')
        logger.info(f'Use transliteration for {lang_name}? -> {do_translit}')

        # Stage 1: Train
        if lang_code not in Config.BASELINE_LANGS:
            logger.info(f'Lang {lang_name} ({lang_code}) doesn`t use baseline model. Train custom!')
            model = _train_data(lang_code=lang_code,
                                lang_translate=lang_translate,
                                lang_speaker=lang_speaker,
                                do_translit=do_translit,
                                lang_ipa=lang_ipa) 
        else:
            logger.info(f'Lang {lang_name} ({lang_code}) is use baseline model')

            if os.path.exists(Config.DEFAULT_MODEL_PATH):
                logger.info(f'Baseline model already exist. Downloading ...')
                with open(Config.DEFAULT_MODEL_PATH, 'rb') as f_model:
                    model = pkl.load(f_model)
            else:
                logger.info(f'Baseline model doesn`t exist. Training ...')
                baseline_row = LANG_DICT[LANG_DICT['language_code'] == Config.DEFAULT_MODEL_LANG_CODE]
                model = _train_data(lang_code=baseline_row['language_code'].values[0],
                                lang_translate=baseline_row['language_translate'].values[0],
                                lang_speaker=baseline_row['google_speaker'].values[0],
                                do_translit=False,
                                lang_ipa=baseline_row['language_ipa'].values[0])

        # Stage 2: Evaluate
        logger.info(f'Evaluating model for {lang_name} ({lang_code})')
        eval_df = _eval_data(model=model,
                            lang_name=lang_name,
                            lang_code=lang_code,
                            lang_speaker=lang_speaker,
                            do_translit=do_translit,
                            lang_ipa=lang_ipa,
                            lang_translate=lang_translate)

        # Stage 3: Plot Results
        logger.info(f'Plotting evaluating result ...')
        _plot_data(eval_df)
        
        # Checking whether the number of used symbols does not exceed 
        # the number of free TTS symbols 
        logger.info(f'Train, Evaluate and Plot processes for {lang_name} -> done!')
        logger.info(f'Spend characters for TTS: {_CHARACTERS_COUNTER}')
        if int(_CHARACTERS_COUNTER) >= (_FREE_TTS_CHARACTERS - 200_000):
            decision = input('Ran out of free symbols for TTS. Continue? y/N')
            if decision.lower() == 'y':
                continue
            else:
                break
        logger.info('-------------------------------------------------------------')

    logger.info(f"Automate training successfully finished!")

if __name__ == "__main__":
    main()