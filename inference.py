import torch
# import sklearn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D

from network import SpeakerEncoder
from utils import cutstr, random_sample, random_cut
import hparams as hp

import numpy as np
import random
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def inference(model, mel_batch):
    # ============================== #
    # input: (batch, length, n_mels) #
    # ============================== #

    with torch.no_grad():
        speaker_embeddings = model(mel_batch)

    return speaker_embeddings


def gen_testdata_for_trainsamples(speaker_id, test_num, dataset_path=hp.dataset_path):
    listdir = os.listdir(dataset_path)
    target_file = []
    for file_name in listdir:
        if cutstr(file_name)[0] == speaker_id:
            target_file.append(file_name)

    target_file = random_sample(target_file, test_num)
    target_mel = list()

    for file_name in target_file:
        file = os.path.join(dataset_path, file_name)
        mel_numpy = np.load(file).T
        mel_numpy = random_cut(mel_numpy)
        target_mel.append(mel_numpy)

    target_mel = np.stack(target_mel)
    # print(np.shape(target_mel))
    # print(type(target_mel))
    batch = torch.from_numpy(target_mel).float().to(device)
    # batch = torch.from_numpy(target_mel)

    return batch


def gen_testdata_for_testsamples(speaker_id, test_num, dataset_path=hp.dataset_test_path):
    listdir = os.listdir(dataset_path)
    target_file = []
    for file_name in listdir:
        if cutstr(file_name)[0] == speaker_id:
            target_file.append(file_name)

    target_file = random_sample(target_file, test_num)
    target_mel = list()

    for file_name in target_file:
        file = os.path.join(dataset_path, file_name)
        mel_numpy = np.load(file).T
        mel_numpy = random_cut(mel_numpy)
        target_mel.append(mel_numpy)

    target_mel = np.stack(target_mel)
    # print(np.shape(target_mel))
    # print(type(target_mel))
    batch = torch.from_numpy(target_mel).float().to(device)
    # batch = torch.from_numpy(target_mel)

    return batch


def test(speaker_id, num, model, test_mode=False):
    if test_mode:
        batch = gen_testdata_for_testsamples(speaker_id, num)
    else:
        batch = gen_testdata_for_trainsamples(speaker_id, num)
    embeddings = inference(model, batch)

    return embeddings


def get_embeddings(speaker_ids, num, model, test_mode=False):
    embeddings = list()
    for i in speaker_ids:
        if test_mode:
            embedding = test(i, num, model, test_mode=True).cpu().numpy()
        else:
            embedding = test(i, num, model).cpu().numpy()
        embeddings.append(embedding)
    embeddings = np.stack(embeddings)
    embeddings = np.reshape(embeddings, (-1, np.shape(embeddings)[2]))

    return embeddings


def pca(embeddings, dim=3):
    pac_model = PCA(dim)
    pca_embeddings = pac_model.fit_transform(embeddings)

    return pca_embeddings


def draw_pic(pca_embeddings, speaker_len, utter_len):
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0, 1, speaker_len)]
    ax = plt.subplot(111, projection='3d')

    for ind, ele in enumerate(pca_embeddings):
        ax.scatter(ele[0], ele[1], ele[2], color=colors[ind // utter_len])

    plt.suptitle("RESULT")
    plt.title("Speaker Embedding")

    ax.set_zlabel('Z')
    ax.set_ylabel('Y')
    ax.set_xlabel('X')

    # plt.show()
    plt.savefig("3d.jpg")


def draw_pic_2D(embeddings, speaker_len, utter_len):
    pca_embeddings = pca(embeddings, 2)
    cmap = plt.get_cmap('viridis')
    colors = [cmap(i) for i in np.linspace(0, 1, speaker_len)]

    plt.figure()
    for ind, ele in enumerate(pca_embeddings):
        plt.scatter(ele[0], ele[1], color=colors[ind // utter_len])
    # plt.show()
    plt.savefig("2d.jpg")


if __name__ == "__main__":

    # Define model
    model = SpeakerEncoder().to(device)
    # model = SpeakerEncoder()
    model.eval()
    print("Model Have Been Defined")

    # Load checkpoint
    checkpoint = torch.load(os.path.join(
        hp.checkpoint_path, 'checkpoint_6000.pth.tar'))
    model.load_state_dict(checkpoint['model'])
    print("Load Done")

    # # Test
    # print(gen_testdata_for_trainsamples(0, 12).size())

    # print(test(0, 12, model).size())
    # print(test(0, 12, model).cpu().numpy())
    # print(np.shape(get_embeddings([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 12, model)))
    # embeddings = get_embeddings([12, 13, 14, 15, 16], 100, model)
    # print(np.shape(pca(embeddings)))
    # print(pca(embeddings))
    # pca_embeddings = pca(embeddings)

    # draw_pic(pca_embeddings, 5, 100)
    # draw_pic_2D(embeddings, 5, 100)

    embeddings = get_embeddings(
        [53, 54, 92, 93, 16], 100, model, test_mode=True)
    pca_embeddings = pca(embeddings)
    draw_pic(pca_embeddings, 5, 100)
    draw_pic_2D(embeddings, 5, 100)
