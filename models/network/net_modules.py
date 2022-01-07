import torch.nn as nn
import torch
import torch.nn.functional as F
import ipdb

class ParserNet(nn.Module):
    def __init__(self, opt, input_size, output_size, batch_size=4, use_gpu=0, layer='UpperClothes', activation='relu',
                 dropout=0.3, output_layer=None):
        super(ParserNet, self).__init__()
        net = [
            nn.Flatten(),
            nn.Linear(input_size, int(input_size / 200)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(int(input_size / 200), int(input_size / 50)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(int(input_size / 50), int(output_size )),
            #nn.ReLU(inplace=True),
            #nn.Linear(int(output_size / 30), output_size),
            #nn.Softmax()

        ]
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)



class SizerNet(nn.Module):
    def __init__(self, opt, input_size, latent_size,  output_size, batch_size=4, use_gpu=0,layer='UpperClothes', activation='relu', dropout=0.3, output_layer=None):
        super(SizerNet, self).__init__()
        self.output_size = output_size
        enc = [
            nn.Flatten(),
            nn.Linear(input_size, int(input_size / 30)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(int(input_size / 30), int(latent_size *10)),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=dropout),
            # nn.Linear(int(input_size / 50), int(latent_size / 10)),
            # nn.ReLU(inplace=True),
            nn.Linear(int(latent_size *10), latent_size),

        ]
        self.enc = nn.Sequential(*enc)
        latent_size2 = latent_size + 10 + 8
        dec = [
            #nn.Flatten(),
            nn.Linear(latent_size2, int(latent_size2 * 50)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(int(latent_size2 * 50), int(output_size / 30)),
            nn.ReLU(inplace=True),
            nn.Linear(int(output_size / 30), output_size),
            #nn.Sigmoid()

        ]
        self.dec = nn.Sequential(*dec)

    def forward(self, x, feat0, feat1, beta_feat):
        enc_out = self.enc(x)
        #append featres
        feat = torch.cat((enc_out, feat0, feat1, beta_feat),1)
        out_dis =  self.dec(feat).view(feat0.shape[0], int(self.output_size/3), 3)
        return out_dis
