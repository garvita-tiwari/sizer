import torch.nn as nn
import torch
import torch.nn.functional as F
import ipdb
class FC_correspondence_lres(nn.Module):
    def __init__(self, input_size, output_size, batch_size=4, use_gpu=0,layer='UpperClothes', activation='relu', dropout=0.3, output_layer=None):
        super(FC_correspondence_lres, self).__init__()
        net = [
            nn.Flatten(),
            nn.Linear(input_size, int(input_size / 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(int(input_size / 3), int(input_size / 50)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(int(input_size / 50), int(output_size / 10)),
            nn.ReLU(inplace=True),
            nn.Linear(int(output_size / 10), output_size),
            #nn.Sigmoid()

        ]
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)
