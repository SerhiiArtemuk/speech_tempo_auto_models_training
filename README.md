# **Speech tempo models training** 

## **Train and test sets**
To carry out training and evaluation, it is necessary to place the `train` and `test` sets of the corresponding language in the `dataset` folder. For example, if the Chineese is trained then this folder will be `dataset/test-set/zh-CN.txt` and `dataset/train-set/zh-CN.txt`. If current language is use `baseline model` then instead of `zh-CN.txt` you need to use `en-US.txt` because this is default language of the `baseline model`.

## **Look up table**
The script uses a `look_up.csv` to get all the necessary information about the language (its code for synthesis, for IPA, for transliteration, speaker code for TTS, whether transliteration is required). Therefore, you need to make sure that the key of the language you are going to use is in this table. If not, this information must be added.

## **Configs set up**
Before running the script, you need to set up the configurations. They are located in the `config.py` file. The setting is as follows:

+ First, you need to set the `TTS_API_KEY`. It is necessary to access Google TTS service.
+ Next, in the variable `LANGUAGE_TO_TRAIN`, place the keys of the languages for which the model needs to be trained.
+ If you are using a language that requires `baseline model `, then the key of this language must also be placed in variable `BASELINE_LANGS`. If the language key is placed in variable `BASELINE_LANGS`, then `baseline model` is used for it, if it already exists in `models`, otherwise `baseline model` is trained.

## **Run script**

***WARNING:*** before running the script, you need to clear the required folders for the language you are using for training. This is: `data/evaluate_backup/audio/{lang_code}` and `data/train_backup/audio/{lang_code}`. For example, if the Chineese (zh-CN) is trained then this folder would be: `data/evaluate_backup/audio/zn-CH` and `data/train_backup/audio/zn-CH`. These folders do not exist by default. They are created during the first training of a specific language. If you need to retrain a language that you have already trained before, you must delete these folders, otherwise the script will use the already generated audio files from these folders and will not send them to Google TTS for synthesizing. This is done so that in case of an error (Сonnection error) it is possible to continue work and to generate the necessary files for training the model.

To run script for auto training use:
```
python train_models_auto.py
```

## **Results**
Script saves all necessary information about train and evaluate.
+ Models are stored in a folder `models/linear_regression_{lang_code}/model.pkl`. All necessary data generated during training (audio files, transcriptions, number of phonemes) are stored in a folder `data/train_вdata_backup/`.
+ The model evaluation results and all generated data are stored in a folder `data/evaluate_backup/`.
+ All created graphs are saved in a folder `plots/`.

**Note:** all these folders are created automatically.