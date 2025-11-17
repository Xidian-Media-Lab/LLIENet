import torch
import torch.nn.functional as F
import torch.nn as nn
from codes.vgg import Vgg16
from args import args
from color_loss import Blur, ColorLoss
def gradient(input_tensor, direction):
    smooth_kernel_x = torch.FloatTensor([[0, 0], [-1, 1]]).view((1, 1, 2, 2)).cuda(args.gpu)
    smooth_kernel_y = torch.transpose(smooth_kernel_x, 2, 3)

    if direction == "x":
        kernel = smooth_kernel_x
    elif direction == "y":
        kernel = smooth_kernel_y
    grad_out = torch.abs(F.conv2d(input_tensor, kernel,
                                  stride=1, padding=1))
    return grad_out

def cct(out):
    L = []
    for i in range(out.shape[0]):
        tp = torch.unsqueeze(out[i, 0, :, :], 0)
        L.append(torch.cat((tp, tp, tp), 0))

    return torch.stack(L,0)

def ave_gradient(input_tensor, direction):
    return F.avg_pool2d(gradient(input_tensor, direction),
                        kernel_size=3, stride=1, padding=1)


def LOSS_coloper(gt, rgb):
    rmean = (rgb[:, 0, :, :] + gt[:, 0, :, :]) / 2
    r = rgb[:, 0, :, :] - gt[:, 0, :, :]
    g = rgb[:, 1, :, :] - gt[:, 1, :, :]
    b = rgb[:, 2, :, :] - gt[:, 2, :, :]
    loss_coloper = (((512 + rmean) * r * r) / 256 + 4 * g * g + ((767 - rmean) * b * b) / 256) ** 0.5
    loss_coloper = torch.mean(loss_coloper)
    return loss_coloper

def Cosineloss(pred,gt):
    func = nn.CosineEmbeddingLoss()
    target = torch.tensor([[[[1]]]], dtype=torch.float).cuda()
    value =func(pred,gt,target)
    return value

def loss_huber(target,input,size_average=None,delta=0.5,reduce=None,reduction='mean'):
    t = torch.abs(input-target)
    ret = torch.where(t <= delta,0.5*(t**2),((delta*t) - (delta**2)/2))
    if reduction != 'none':
        ret = torch.mean(ret) if reduction == 'mean' else torch.sum(ret)
    return ret

def smooth(input_I, input_R):
    # input_R = 0.299 * input_R[:, 0, :, :] + 0.587 * input_R[:, 1, :, :] + 0.114 * input_R[:, 2, :, :]
    # input_R = torch.unsqueeze(input_R, dim=1)
    return torch.mean(gradient(input_I, "x") * torch.exp(-10 * ave_gradient(input_R, "x")) +
                      gradient(input_I, "y") * torch.exp(-10 * ave_gradient(input_R, "y")))
def loss_pre(pred_hs,input_high_hs,pred,gt):
    losshuber = loss_huber(input_high_hs,pred_hs)

    loss_color = ColorLoss().cuda()
    blur = Blur(3).cuda()
    blur_rgb = blur(pred)
    blur_gt = blur(gt)
    loss_color = loss_color(blur_rgb, blur_gt)
    return losshuber + loss_color
def computeloss(input_low,input_high,R_low, I_low,R_high, I_high,rgb, predy,predcbcr,gty,gtcbcr,gt):
    # R_low, I_low = DecomNet(input_low)
    # R_high, I_high = DecomNet(input_high)


    # Other variables
    # I_low_3 = torch.cat((I_low, I_low, I_low), dim=1)
    # I_high_3 = torch.cat((I_high, I_high, I_high), dim=1)

    # Compute losses
    recon_loss_low = F.l1_loss(R_low * I_low, input_low)
    recon_loss_high = F.l1_loss(R_high * I_high, input_high)
    recon_loss_mutal_low = F.l1_loss(R_high * I_low, input_low)
    recon_loss_mutal_high = F.l1_loss(R_low * I_high, input_high)
    equal_R_loss = F.l1_loss(R_low, R_high.detach())

    Ismooth_loss_low = smooth(I_low, R_low)
    Ismooth_loss_high = smooth(I_high, R_high)

    loss_Decom = recon_loss_low + \
                      recon_loss_high + \
                      0.001 * recon_loss_mutal_low + \
                      0.001 * recon_loss_mutal_high + \
                      0.1 * Ismooth_loss_low + \
                      0.1 * Ismooth_loss_high + \
                      0.01 * equal_R_loss
    criteron = nn.MSELoss()
    criteron1 = nn.L1Loss()
    loss_mse_func = nn.MSELoss()
    vgg = Vgg16(requires_grad=False).cuda(args.gpu)
    ou1 = vgg(cct(predy)) 
    gd1 = vgg(cct(gty))
    contentloss = criteron(ou1.relu3_3, gd1.relu3_3) + criteron(ou1.relu4_3, gd1.relu4_3) \
                  + criteron(ou1.relu1_2, gd1.relu1_2) + criteron(ou1.relu2_2, gd1.relu2_2)

    loss2 = criteron1(predy, gty)
    # loss3 = criteron1(predcbcr, gtcbcr)
    # loss3_2 = Cosineloss(predcbcr,gtcbcr)
    # loss4 = criteron1(output1_1[3], ycb) + criteron1(output1_1[4], ycr)
    losshuber = loss_huber(gtcbcr,predcbcr)
    lossfunc = nn.L1Loss()


    loss_colorrecovery = loss2 + contentloss

    # rmean = (rgb[:,0,:,:] + gt[:,0,:,:])/2
    # r = rgb[:, :, 0] - gt[:, :, 0]
    # g = rgb[:, :, 1] - gt[:, :, 1]
    # b = rgb[:, :, 2] - gt[:, :, 2]
    # loss_coloper = (((512+rmean)*r*r)/256 + 4*g*g + ((767-rmean)*b*b)/256) ** 0.5

    loss_color = ColorLoss().cuda()
    blur = Blur(3).cuda()
    blur_rgb = blur(rgb)
    blur_gt = blur(gt)
    loss_color = loss_color(blur_rgb,blur_gt)



    return loss_colorrecovery + loss_Decom + losshuber + loss_color

if __name__ == '__main__':
    A = torch.randn(4,3,5,5)
    B = torch.randn(4,3,5,5)
    loss = LOSS_coloper(A,B)
    loss = torch.mean(loss)
    print(loss)