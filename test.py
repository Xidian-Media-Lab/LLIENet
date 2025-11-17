import glob
from LLIENet import LLINet
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import os
import sys
# from torchvision.transforms import
from torchvision.transforms import ToPILImage
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from tqdm import tqdm
from cwan.cwannet.utils.rgb2lab import LAB
from cwan.cwannet.net.cwan_net import CWAN
from args import args
from train import RGB_ycbcr
# parser = argparse.ArgumentParser(description='')
#
# parser.add_argument('--gpu_id', dest='gpu_id',
#                     default="0",
#                     help='GPU ID (-1 for CPU)')
# parser.add_argument('--data_dir', dest='data_dir',
#                     default='./LOLdataset/eval15/low/',
#                     help='directory storing the test data')
# parser.add_argument('--ckpt_dir', dest='ckpt_dir',
#                     default='./modles/',
#                     help='directory for checkpoints')
# parser.add_argument('--res_dir', dest='res_dir',
#                     default='./results/test/low/',
#                     help='directory for saving the results')
#
# args = parser.parse_args()
#
#
# def predict(self,
#             test_low_data_names,
#             res_dir,
#             ckpt_dir):
#     # Load the network with a pre-trained checkpoint
#     self.train_phase = 'DecomNet'
#     load_model_status, _ = self.load(ckpt_dir)
#     if load_model_status:
#         print(self.train_phase, "  : Model restore success!")
#     else:
#         print("No pretrained model to restore!")
#         raise Exception
#     self.train_phase = 'Mynet'
#     load_model_status, _ = self.load(ckpt_dir)
#     if load_model_status:
#         print(self.train_phase, ": Model restore success!")
#     else:
#         print("No pretrained model to restore!")
#         raise Exception
#
#     # Set this switch to True to also save the reflectance and shading maps
#     save_R_L = False
#
#     # Predict for the test images
#     for idx in range(len(test_low_data_names)):
#         test_img_path = test_low_data_names[idx]
#         test_img_name = test_img_path.split('/')[-1]
#         print('Processing ', test_img_name)
#         test_low_img = Image.open(test_img_path)
#         test_low_img = np.array(test_low_img, dtype="float32") / 255.0
#         test_low_img = np.transpose(test_low_img, (2, 0, 1))
#         input_low_test = np.expand_dims(test_low_img, axis=0)
#
#         self.forward(input_low_test, input_low_test)
#         result_1 = self.output_R_low
#         result_2 = self.output_I_low
#         result_3 = self.output_I_delta
#         result_4 = self.output_S
#         input = np.squeeze(input_low_test)
#         result_1 = np.squeeze(result_1)
#         result_2 = np.squeeze(result_2)
#         result_3 = np.squeeze(result_3)
#         result_4 = np.squeeze(result_4)
#
#         if save_R_L:
#             cat_image = np.concatenate([input, result_1, result_2, result_3, result_4], axis=2)
#         else:
#             cat_image = np.concatenate([input, result_4], axis=2)
#
#         cat_image = np.transpose(cat_image, (1, 2, 0))
#         result_4 = np.transpose(result_4.numpy(), (1, 2, 0))
#         # print(cat_image.shape)
#         im = Image.fromarray(np.clip(cat_image * 255.0, 0, 255.0).astype('uint8'))
#         im2 = Image.fromarray(np.clip(result_4 * 255.0, 0, 255.0).astype('uint8'))
#         filepath = res_dir + '/' + test_img_name
#         im.save(filepath[:-4] + '.jpg')
#         im2.save(filepath[:-4] + '.only.jpg')
#
# def pre(model):
#
#     test_low_data_names = glob(args.data_dir + '/' + '*.*')
#     test_low_data_names.sort()
#     print('Number of evaluation images: %d' % len(test_low_data_names))
#
#     predict(test_low_data_names,
#                 res_dir=args.res_dir,
#                 ckpt_dir=args.ckpt_dir)
#
#
# if __name__ == '__main__':
#     if args.gpu_id != "-1":
#         # Create directories for saving the results
#         if not os.path.exists(args.res_dir):
#             os.makedirs(args.res_dir)
#         # Setup the CUDA env
#         os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
#         # Create the model
#         model = LLINet().cuda()
#         # Test the model
#         pre(model)
#     else:
#         # CPU mode not supported at the moment!
#         raise NotImplementedError

class MyTestDataset(Dataset):
    def __init__(self, img1_path_file):
        exts = ("*.png","*.PNG","*.jpg","*.JPG", "*.jpeg","*.bmp","*.tif", "*.tiff")
        self.img_list = []
        for ext in exts:
            self.img_list += glob.glob(os.path.join(img1_path_file,ext))
        self.img_list.sort()

    def __getitem__(self, index):
        img_path = self.img_list[index]
        # img = Image.open(self.img_list[index]).convert('YCbCr')
        # img = np.array(img)
        img = cv2.imread(img_path)
        im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        custom_transform= transforms.Compose([transforms.ToTensor()])
        rgb_image = custom_transform(im)

        filename = os.path.basename(img_path)
        return rgb_image, filename

    def __len__(self):
        return len(self.img_list)
def showtensor(im,i,label):
    # im = im.cpu().squeeze()
    # im = ToPILImage()(im)
    rgb_output = im
    print(rgb_output.max().detach())
    print(rgb_output.min().detach())
    im = rgb_output[0].cpu().detach().numpy().transpose(1, 2, 0)
    im = im *255
    # cv2.imshow('reslt',im)
    # cv2.waitKey(0)
    im = ToPILImage()(im.astype('uint8'))
    # plt.show()
    # im.show()
    im.save(args.test_save_dir + '{}_pred_ssim99_{}.png'.format(str(i).zfill(2),label))
    print('finish')
if __name__ == '__main__':

    model = LLINet()
    # cwan = CWAN().eval().cuda()
    # cwan.cwan_ab.load_state_dict(torch.load('./models/ckpt_CWAB_30.pt'))
    is_cuda = torch.cuda.is_available()
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint['model_state_dict'])

    # img1_path_file = args.test_visible
    img2_path_file = args.test_lwir

    testloader = DataLoader(MyTestDataset(img2_path_file), batch_size=1,
                            num_workers=1)
    rgb_YCBCR = RGB_ycbcr()

    model.cuda()

    for i, (data, filenames) in enumerate(tqdm(testloader)):
        # print(data.shape)
        # data /= 255.
        low_hsv = rgb_YCBCR.rgb_to_ycbcr(data)
        b, c, h, w = low_hsv.size()
        low_hsv = F.interpolate(low_hsv, size=(400,600), mode = 'bilinear', align_corners=False)
        img_v,img_hs = low_hsv[:, 0].view(1,1,400,600), low_hsv[:, 1:]
        if is_cuda and args.gpu is not None:
            img_v = img_v.cuda()
            img_hs = img_hs.cuda()


        R_y,I_y = model.DecomNet(img_v)
        # showtensor(R_y,i,'R_y')
        # showtensor(I_y,i,'I_y')

        pred_I, pred_hs = model.colorrecovery(I_y, img_hs)
        # pred_ab2 = cwan(img_ab)

        pred_v = torch.mul(R_y, pred_I)
        # showtensor(pred_y,i,'pred_y')

        pre = torch.cat([pred_v,pred_hs],dim=1)
        pre_rgb = rgb_YCBCR.yccbr_to_rgb(pre)
        pre_rgb = F.interpolate(pre_rgb, size=(h,w), mode = 'bilinear', align_corners=False)

        # filenames = args.test_save_dir + '{}_pred_ycb99_{}.png'.format(str(i).zfill(2),'hsv')
        save_path = os.path.join(args.test_save_dir, filenames[0])


        save_image(pre_rgb,save_path)



        print('Finished testing')
