import torch
import torch.nn as nn
import torch.nn.functional as F
from cwan.cwannet.net.cwan_parts import CWAN_AB
from cwan.cwannet.utils.rgb2lab import LAB
from args import args

class DecomNet(nn.Module):
    def __init__(self, channel=64, kernel_size=3):
        super(DecomNet, self).__init__()
        # Shallow feature extraction
        self.net1_conv0 = nn.Conv2d(2, channel, kernel_size*3,
                                    padding=4, padding_mode='replicate')
        # Activated layers!
        self.net1_convs = nn.Sequential(nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU(),
                                        nn.Conv2d(channel, channel, kernel_size,
                                                  padding=1, padding_mode='replicate'),
                                        nn.ReLU())
        # Final recon layer
        self.net1_recon = nn.Conv2d(channel, 2, kernel_size,
                                    padding=1, padding_mode='replicate')

    def forward(self, input_im):
        # B,C,H,W = input_im.shape[0],input_im.shape[1],input_im.shape[2],input_im.shape[3]
        input_max= torch.max(input_im, dim=1, keepdim=True)[0]
        input_img= torch.cat((input_max, input_im), dim=1)
        feats0   = self.net1_conv0(input_img)
        featss   = self.net1_convs(feats0)
        outs     = self.net1_recon(featss)
        B, C, H, W = outs.shape[0], outs.shape[1], outs.shape[2], outs.shape[3]
        R        = torch.sigmoid(outs[:, 0, :, :]).view(B,1,H,W)
        L        = torch.sigmoid(outs[:, 1, :, :]).view(B,1,H,W)
        return R, L
class Myblock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, B_I=False, bn=True):
        super(Myblock, self).__init__()
        self.bn = bn
        self.relu = nn.LeakyReLU(0.1)
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride,
                               kernel_size=3, padding=1, bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, stride=stride,
                               kernel_size=3, padding=1, bias=False)

        self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, stride=stride,
                               kernel_size=1, padding=0, bias=False)

        self.conv4 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, stride=stride,
                               kernel_size=3, padding=1, bias=False)

        self.conv5 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, stride=stride,
                               kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        temp = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.conv3(temp) + x
        x = self.relu(x)
        temp = x
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x + temp)
        return x

class YSUBNet(nn.Module):
    def __init__(self, block=Myblock):
        super(YSUBNet, self).__init__()
        self.Y_l1 = block(in_channels=1, out_channels=64)
        # half_size
        self.Y_down1 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, stride=2, kernel_size=3, padding=1),
                                     nn.LeakyReLU(0.1))

        self.Y_l2 = block(in_channels=128, out_channels=128)

        self.Y_down2 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, stride=2, kernel_size=3, padding=1),
                                     nn.LeakyReLU(0.1))

        self.Y_l3 = block(in_channels=256, out_channels=128)

        ##### half_size


        self.Y_up2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, stride=2, kernel_size=4, padding=1),
            nn.LeakyReLU(0.1))


        self.Y_l6 = block(in_channels=64, out_channels=64)
        ###### origin_size
        self.Y_up3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, stride=2, kernel_size=4, padding=1),
            nn.LeakyReLU(0.1))


        self.Y_l7 = block(in_channels=64, out_channels=64)

        self.Y_l8 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=32, stride=1, kernel_size=1),
                                  nn.LeakyReLU(0.1))

        self.Y_out = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=1, stride=1, kernel_size=3, padding=1))

    def forward(self, y):
        x = self.Y_l1(y)
        x = self.Y_down1(x)
        x = self.Y_l2(x)
        x = self.Y_down2(x)
        x = self.Y_l3(x)
        x = self.Y_up2(x)
        x = self.Y_l6(x)
        x = self.Y_up3(x)
        x = self.Y_l7(x)
        x = self.Y_l8(x)
        x = self.Y_out(x)

        return x

class CbcrSUBNet(nn.Module):
    def __init__(self):
        super(CbcrSUBNet, self).__init__()
        self.C_l1 = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=64, stride=1, kernel_size=3, padding=1),
                                  nn.LeakyReLU(0.1))
        self.C_down1 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, stride=2, kernel_size=3, padding=1),
                                     nn.LeakyReLU(0.1))
        self.C_l2 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, stride=1, kernel_size=3, padding=1),
                                  nn.LeakyReLU(0.1))
        self.C_down2 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, stride=2, kernel_size=3, padding=1),
                                     nn.LeakyReLU(0.1))
        self.C_l3 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=128, stride=1, kernel_size=3, padding=1),
                                  nn.LeakyReLU(0.1))
        self.C_up1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, stride=2, kernel_size=4, padding=1),
            nn.LeakyReLU(0.1))

        self.C_l6 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, stride=1, kernel_size=3, padding=1),
                                  nn.LeakyReLU(0.1))
        self.C_up2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, stride=2, kernel_size=4, padding=1),
            nn.LeakyReLU(0.1))
        self.C_l7 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=32, stride=1, kernel_size=3, padding=1),
                                  nn.LeakyReLU(0.1))

        self.C_l8 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=16, stride=1, kernel_size=1),
                                  nn.LeakyReLU(0.1))
        self.C_out = nn.Sequential(nn.Conv2d(in_channels=16, out_channels=2, stride=1, kernel_size=3, padding=1))
        # self.covercbcr = CWAN_AB()

    def forward(self, cbcr):
        c = self.C_l1(cbcr)
        c = self.C_down1(c)
        c = self.C_l2(c)
        c = self.C_down2(c)
        c = self.C_l3(c)
        c = self.C_up1(c)
        c = self.C_l6(c)
        c = self.C_up2(c)
        c = self.C_l7(c)
        c = self.C_l8(c)
        c = self.C_out(c)
        return c

class Mynet(nn.Module):
    '''
    model name: S*.pkl
    '''
    def __init__(self, block=Myblock):
        super(Mynet, self).__init__()
        self.g_y = YSUBNet()
        self.g_cbcr = CbcrSUBNet()
        checkpoint = torch.load(args.resume_pre)
        self.g_cbcr.load_state_dict(checkpoint['model_state_dict'])
        print('load pre-trained model')

    def forward(self, y, cbcr):
        y = self.g_y(y)
        c = self.g_cbcr(cbcr)
        return y, c

class LLINet(nn.Module):
    def __init__(self):
        super(LLINet, self).__init__()
        self.DecomNet = DecomNet()
        self.colorrecovery =Mynet()
    def forward(self, input_high_y, input_low_y,input_low_ab):
        # high = self.lab_converter(input_high)
        # input_high_y, input_high_ab = high[:, :1], high[:, 1:]
        #
        # low = self.lab_converter(input_low)
        # input_low_y, input_low_ab = low[:, :1], low[:, 1:]
        R_high, I_high = self.DecomNet(input_high_y)
        R_low, I_low = self.DecomNet(input_low_y)
        pred_y,pred_cbcr = self.colorrecovery(I_low,input_low_ab)
        pred_y = torch.mul(R_low,pred_y)
        return R_high, R_low, I_high, I_low, pred_y, pred_cbcr

if __name__ == "__main__":
    a = torch.rand(2,1,16,16)
    b = torch.rand(2,1,16,16)
    c = torch.rand(2,2,16,16)
    model = LLINet()
    R_high, R_low, I_high, I_low, pred_y, pred_cbcr = model(a,b,c)
    print('finish')