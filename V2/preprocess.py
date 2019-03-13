import os
import threading
from tqdm import tqdm
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from functools import partial

import audio
import hparams as hp


def preprocess(speaker_id, speaker_file_name, class_num_remaining=50, data_path=hp.origin_data):
    out_dataset = hp.dataset_path
    if not os.path.exists(out_dataset):
        os.mkdir(out_dataset)

    file_path = os.path.join(data_path, speaker_file_name)
    total_len = len(os.listdir(file_path))
    cut_length = total_len - class_num_remaining
    wav_file_list = os.listdir(file_path)[0:cut_length]

    for utterance_id, wav_file in enumerate(wav_file_list):
        wav_file_path = os.path.join(file_path, wav_file)
        wav = audio.load_wav(wav_file_path)
        mel_spec = audio.melspectrogram(wav)

        save_file_name = str(speaker_id) + "_" + str(utterance_id) + ".npy"
        np.save(os.path.join(out_dataset, save_file_name), mel_spec)


def preprocess_test(speaker_id, speaker_file_name, class_num_remaining=50, data_path=hp.origin_data):
    out_dataset = hp.dataset_test_path
    if not os.path.exists(out_dataset):
        os.mkdir(out_dataset)

    file_path = os.path.join(data_path, speaker_file_name)
    total_len = len(os.listdir(file_path))
    cut_length = total_len - class_num_remaining
    wav_file_list = os.listdir(file_path)[cut_length:]

    for utterance_id, wav_file in enumerate(wav_file_list):
        wav_file_path = os.path.join(file_path, wav_file)
        wav = audio.load_wav(wav_file_path)
        mel_spec = audio.melspectrogram(wav)

        save_file_name = str(speaker_id) + "_" + str(utterance_id) + ".npy"
        np.save(os.path.join(out_dataset, save_file_name), mel_spec)


if __name__ == "__main__":

    # # preprocess(0, "p225")
    # list_speaker = os.listdir(hp.origin_data)
    # # thrs = [threading.Thread(target=preprocess, args=[speaker_id, file_name])
    # #         for speaker_id, file_name in enumerate(list_speaker)]
    # # [thr.start() for thr in thrs]
    # # [thr.join() for thr in thrs]

    # executor = ProcessPoolExecutor(max_workers=cpu_count())
    # futures = [executor.submit(partial(preprocess, speaker_id, file_name))
    #            for speaker_id, file_name in enumerate(list_speaker)]
    # [future.result() for future in futures]

    list_speaker = os.listdir(hp.origin_data)
    executor = ProcessPoolExecutor(max_workers=cpu_count())
    futures = [executor.submit(partial(preprocess_test, speaker_id, file_name))
               for speaker_id, file_name in enumerate(list_speaker)]
    [future.result() for future in futures]
