import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import random
from multiprocessing import cpu_count

import hparams


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SVData(Dataset):
    """VCTK"""

    def __init__(self, dataset_path=hparams.dataset_path):
        self.dataset_path = dataset_path

    def __len__(self):
        list_dir = os.listdir(self.dataset_path)
        # print(len(list_dir))
        # speaker_len = int(list_dir[len(list_dir)-1][0]) + 1
        # print(speaker_len)
        speaker_len = 109

        return speaker_len

    def random_sample(self, total, sample_length):
        out = [i for i in range(total)]
        return random.sample(out, sample_length)

    def random_cut(self, mel, length):
        total_length = np.shape(mel)[0]

        if total_length <= hparams.tisv_frame:
            # raise ValueError("total length is too short!")
            mel = np.concatenate((mel, mel[0:total_length, :]))

        total_length = np.shape(mel)[0]

        if total_length < 181:
            # raise ValueError("something wrong!")
            mel = np.concatenate((mel, mel[0:total_length, :]))

        total_length = np.shape(mel)[0]

        if total_length < 181:
            # raise ValueError("something wrong!")
            mel = np.concatenate((mel, mel[0:total_length, :]))
        
        total_length = np.shape(mel)[0]

        start = random.randint(0, total_length - length - 1)

        return mel[start:start + length, :]

    def process_mel(self, name):
        mel_spec = np.load(name)
        mel_spec = np.transpose(mel_spec)
        mel_spec = self.random_cut(mel_spec, hparams.tisv_frame)

        return mel_spec

    def __getitem__(self, index):
        list_file_name = [str(index) + "_" + str(ind) +
                          ".npy" for ind in self.random_sample(hparams.total_utterance, hparams.M)]

        out = np.stack([self.process_mel(os.path.join(
            hparams.dataset_path, file_name))for file_name in list_file_name])

        # out = torch.from_numpy(out).to(device)
        # out = out.contiguous()
        # out = out.view(-1, hparams.tisv_frame, hparams.n_mels_channel)

        # out = np.reshape(out, (-1, hparams.tisv_frame, hparams.n_mels_channel))
        # print(np.shape(out))

        return out


def collate_fn(batch):
    mels = [mel for mel in batch]
    mels = np.stack(mels)
    # mels = torch.from_numpy(mels).to(device)
    # mels = mels.contiguous().view(-1, hparams.tisv_frame, hparams.n_mels_channel)

    mels = np.reshape(mels, (-1, hparams.tisv_frame, hparams.n_mels_channel))

    return mels


if __name__ == "__main__":

    dataset = SVData()
    test_loader = DataLoader(
        dataset, batch_size=2, collate_fn=collate_fn, shuffle=True, num_workers=cpu_count())

    print(len(test_loader))
    for index, batch in enumerate(test_loader):
        print(np.shape(batch))
        batch = torch.from_numpy(batch).to(device)
        batch = batch.contiguous().view(-1, hparams.tisv_frame, hparams.n_mels_channel)
        print(batch.size())
