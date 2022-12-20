import torch
from torch import nn
import torch.nn.functional as F
from torchsummary import summary


class UpScailing(nn.Module):

    def __init__(self, scale_factor=2.0):
        super().__init__()

        self.scale_factor = scale_factor

    
    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode="nearest")



class UNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.channel_list = [3, 16, 64, 256, 512]
        encoder_channels = [(3, 16), (16, 64), (64, 128), (128, 256), (256, 512), (512, 512)]
        decoder_channels = [(512, 512), (1024, 512), (1024, 256), (512, 128), (256, 64), (128, 16)]

        def conv_blocks(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(),

                nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU()
            )

        
        self.encoder = nn.ModuleList([
            conv_blocks(in_channels=in_channels, out_channels=out_channels)
            for in_channels, out_channels in encoder_channels
        ])

        self.decoder = nn.ModuleList([
            conv_blocks(in_channels=in_channels, out_channels=out_channels)
            for in_channels, out_channels in decoder_channels
        ])

        self.maxpooling = nn.MaxPool2d(2, 2)
        self.upscailing = UpScailing(scale_factor=2.0)

        self.classifier = nn.Sequential(
            *conv_blocks(in_channels=32, out_channels=3),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    
    def forward(self, x):
        encoder_outs = []
        out = x

        for i in range(len(self.encoder)):
            out = self.encoder[i](out)
            encoder_outs.append(out)
            out = self.maxpooling(out)
        
        for i in range(len(self.decoder)):
            out = self.upscailing(self.decoder[i](out))
            out = torch.cat((out, encoder_outs[-i-1]), dim=1)

        out = self.classifier(out)
        
        return out



if __name__ == '__main__':

    unet = UNet().cuda()

    num_params = sum([p.numel() for p in unet.parameters()])
    print("num_params:", num_params)

    print("output shape:", unet(torch.zeros((2, 3, 256, 256)).cuda()).shape)