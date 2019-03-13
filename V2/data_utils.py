import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
from multiprocessing import cpu_count

import hparams
from utils import cutstr


device = torch.device('cuda'if torch.cuda.is_available()else 'cpu')


class SVData(Dataset):
    """VCTK"""

    def __init__(self, dataset_path = hparams.dataset_path):
        self.dataset_path = dataset_path

    def __len__(self):
        all_num = list()
        for file in os.listdir(self.dataset_path):
            num, _ = cutstr(file)
            all_num.append(num)
        all_num = np.array(all_num)
        speaker_len = all_num.max() + 1

        # print(speaker_len)
        return speaker_len

    def random_sample(self, total, sample_length):
        out = [i for i in range(total)]
        return random.sample(out, sample_length)

    def random_cut(self, mel, length):
        total_length = np.shape(mel)[0]

        if total_length <= hparams.tisv_frame:
            mel = np.concatenate([mel for i in range(10)])
            total_length = np.shape(mel)[0]

        if total_length <= hparams.tisv_frame:
            raise ValueError("total length is too short!")

        start = random.randint(0, total_length - length - 1)

        return mel[start:start + length, :]

    def process_mel(self, name):
        mel_spec = np.load(name)
        mel_spec = np.transpose(mel_spec)
        mel_spec = self.random_cut(mel_spec, hparams.tisv_frame)

        return mel_spec

    def get_total_len(self, index):
        list_dir = os.listdir(self.dataset_path)
        list_utt = list()
        for file in list_dir:
            num1, num2 = cutstr(file)
            if num1 == index:
                list_utt.append(num2)

        return np.array(list_utt).max()

    def __getitem__(self, index):
        total_length = self.get_total_len(index)

        # print(index, total_length)

        list_file_name = [str(index) + "_" + str(ind) + 
                          ".npy" for ind in self.random_sample(total_length, hparams.M)]
        # print(self.random_sample(total_length, hparams.M))

        out = np.stack([self.process_mel(os.path.join(
            hparams.dataset_path, file_name))for file_name in list_file_name])

        # print(np.shape(out))
        return out


def collate_fn(batch):
    mels = [mel for mel in batch]
    mels = np.stack(mels)
    # mels = torch.from_numpy(mels).to(device)
    # mels = mels.contiguous().view(-1, hparams.tisv_frame, hparams.n_mels_channel)

    mels = np.reshape(mels, (-1, hparams.tisv_frame, hparams.num_mels))

    return mels


if __name__ == "__main__":

    dataset = SVData()
    test_loader = DataLoader(dataset, batch_size = hparams.N, collate_fn = collate_fn, 
                             drop_last = True, shuffle = True, num_workers = cpu_count())

    print(len(test_loader))
    for index, batch in enumerate(test_loader):
        print(np.shape(batch))
