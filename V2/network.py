import torch
import torch.nn as nn

from module import PreLinear, ResNet
import hparams as hp


class SpeakerEncoder(nn.Module):
    """Speaker Encoder"""

    def __init__(self):
        super(SpeakerEncoder, self).__init__()

        self.prenet = PreLinear()
        self.lstm = nn.LSTM(hp.pre_output_size,
                            hp.hidden_dim,
                            num_layers=hp.num_layer,
                            batch_first=True,
                            bidirectional=True)
        self.cnn = ResNet()
        self.projection = nn.Linear(self.get_len(), hp.speaker_dim)
        self.init_params()

    def init_params(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

    def get_len(self):
        x = torch.randn(2, 180, 80)
        x = self.prenet(x)
        x, _ = self.lstm(x)
        x_1 = x[:, :, 0:hp.hidden_dim]
        x_2 = x[:, :, hp.hidden_dim:]
        x = x_1 + x_2
        x = x.unsqueeze(1)
        x = self.cnn(x)

        return x.size(1)
    
    def forward(self, x):
        # ============================== #
        # input: (batch, length, n_mels) #
        # ============================== #

        # PreLinear
        # x = torch.stack([self.prenet(x[:, i]) for i in range(x.size(1))], 1)
        x = self.prenet(x)

        # LSTM
        x, _ = self.lstm(x)
        x_1 = x[:, :, 0:hp.hidden_dim]
        x_2 = x[:, :, hp.hidden_dim:]
        x = x_1 + x_2

        # ResNet
        x = x.unsqueeze(1)
        x = self.cnn(x)

        # Linear
        x = self.projection(x)

        # Norm
        x = x / torch.norm(x)

        return x


if __name__ == "__main__":

    test_model = SpeakerEncoder()
    print(test_model)

    test_input = torch.randn(2, 180, 80)
    output = test_model(test_input)
    print(output.size())
