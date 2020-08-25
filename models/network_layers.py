import torch.nn as nn
import torch
import torch.nn.functional as F
import ipdb


class FC_correspondence_hres(nn.Module):
    def __init__(self, input_size, output_size, batch_size=4, use_gpu=0, layer='UpperClothes', activation='relu',
                 dropout=0.3, output_layer=None):
        super(FC_correspondence_hres, self).__init__()
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
class Enc_lres(nn.Module):
    def __init__(self, input_size, output_size, batch_size=4, use_gpu=0,layer='UpperClothes', activation='relu', dropout=0.3, output_layer=None):
        super(Enc_lres, self).__init__()
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

class Dec_lres(nn.Module):
    def __init__(self, input_size, output_size, batch_size=4, use_gpu=0,layer='UpperClothes', activation='relu', dropout=0.3, output_layer=None):
        super(Dec_lres, self).__init__()
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


class EncDec_hres(nn.Module):
    def __init__(self, input_size, latent_size,  output_size, batch_size=4, use_gpu=0,layer='UpperClothes', activation='relu', dropout=0.3, output_layer=None):
        super(EncDec_hres, self).__init__()
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


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)


class Discriminator_size(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator_size, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, mesh_A, mesh_B=None):
        # Concatenate image and condition image by channels to produce input
        #not needed fo sizer if we are using only one kind of ibject, we can use betas or something else here or maybe smpl body
        mesh = mesh_A
        if mesh_B is not None:
            mesh = torch.cat((mesh_A, mesh_B),1)
        return self.model(mesh)
