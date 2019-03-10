import torch
import torch.nn as nn
from torch import optim

from network import SpeakerEncoder
from data_utils import DataLoader, collate_fn
from data_utils import SVData
from loss import GE2ELoss
import hparams as hp

from multiprocessing import cpu_count
import numpy as np
import argparse
import os
import time


cuda_available = torch.cuda.is_available()


def main(args):
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define model
    model = SpeakerEncoder().to(device)
    GE2E_loss = GE2ELoss()
    print("Model Have Been Defined")

    # Get dataset
    dataset = SVData()

    # Optimizer
    optimizer = torch.optim.SGD([
        {'params': model.parameters()},
        {'params': GE2E_loss.parameters()}],
        lr=hp.learning_rate
    )

    # Get training loader
    training_loader = DataLoader(
        dataset, batch_size=hp.N, shuffle=True, collate_fn=collate_fn, num_workers=cpu_count())
    print("Get Training Loader")

    # Load checkpoint if exists
    try:
        checkpoint = torch.load(os.path.join(
            hp.checkpoint_path, 'checkpoint_%d.pth.tar' % args.restore_step))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("\n---Model Restored at Step %d---\n" % args.restore_step)
    except:
        print("\n---Start New Training---\n")
        if not os.path.exists(hp.checkpoint_path):
            os.mkdir(hp.checkpoint_path)

    # Define Some Information
    total_step = hp.epochs * len(training_loader)
    Time = np.array([])
    Start = time.perf_counter()

    # Training
    model = model.train()

    for epoch in range(hp.epochs):
        for i, batch in enumerate(training_loader):
            start_time = time.perf_counter()

            # Count step
            current_step = i + args.restore_step + \
                epoch * len(training_loader) + 1

            # Init
            optimizer.zero_grad()

            # Load Data
            batch = torch.from_numpy(batch).float().to(device)
            embeddings = model(batch)

            # Loss
            embeddings = embeddings.contiguous().view(hp.N, hp.M, -1)
            loss = GE2E_loss(embeddings)

            # Backward
            loss.backward()

            # Clipping gradients to avoid gradient explosion
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            nn.utils.clip_grad_norm_(GE2E_loss.parameters(), 1.0)

            # Update weights
            optimizer.step()

            if current_step % hp.log_step == 0:
                Now = time.perf_counter()
                str_loss = "Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}.".format(
                    epoch + 1, hp.epochs, current_step, total_step, loss.item())
                str_time = "Time Used: {:.3f}s, Estimated Time Remaining: {:.3f}s.".format(
                    (Now - Start), (total_step - current_step) * np.mean(Time))

                print(str_loss)
                print(str_time)

                with open("logger.txt", "a")as f_logger:
                    f_logger.write(str_loss + "\n")
                    f_logger.write(str_time + "\n")
                    f_logger.write("\n")

            if current_step % hp.save_step == 0:
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(
                )}, os.path.join(hp.checkpoint_path, 'checkpoint_%d.pth.tar' % current_step))
                print("\nsave model at step %d ...\n" % current_step)

            end_time = time.perf_counter()
            Time = np.append(Time, end_time - start_time)
            if len(Time) == hp.clear_Time:
                temp_value = np.mean(Time)
                Time = np.delete(
                    Time, [i for i in range(len(Time))], axis=None)
                Time = np.append(Time, temp_value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step', type=int,
                        help='Global step to restore checkpoint', default=0)

    args = parser.parse_args()
    main(args)
