import os
import argparse
import torch
import pdb
import torch.nn as nn

from utils.test_3d_patch import *

from pancreas.Vnet import VNet
from networks.ResVNet import ResVNet

# from testutildtc import *
# from test_usenet.dtc import VNet
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./Datasets/pancreas', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='HAP', help='exp_name')
parser.add_argument('--model', type=str, default='VNet', help='model_name')
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--detail', type=int, default=1, help='print metrics for every samples?')
parser.add_argument('--nms', type=int, default=0, help='apply NMS post-processing?')
parser.add_argument('--labelnum', type=int, default=12, help='labeled data')
parser.add_argument('--stage_name', type=str, default='self_train', help='self_train or pre_train')

FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu
snapshot_path = "./model/HAP/pancreas_{}_{}_labeled/{}".format(FLAGS.exp, FLAGS.labelnum, FLAGS.stage_name)
test_save_path = "./model/HAP/pancreas_{}_{}_labeled/{}_predictions/".format(FLAGS.exp, FLAGS.labelnum, FLAGS.model)
num_classes = 2

if not os.path.exists(test_save_path):
    os.makedirs(test_save_path)
print(test_save_path)
with open(FLAGS.root_path + '/data_split/test.txt', 'r') as f:
    image_list = f.readlines()
image_list = [FLAGS.root_path + "/data/pancreas_data/" + "PANCREAS_" + item.replace('\n', '') for item in
              image_list]


def create_Vnet(ema=False):
    net = VNet(n_channels=1, n_classes=2, normalization='instancenorm')
    net = nn.DataParallel(net)
    model = net.cuda()
    if ema:
        for param in model.parameters():
            param.detach_()
    return model


def testLA():
    net1 = create_Vnet()
    
    ema_net1 = create_Vnet()
    ema_net2 = create_Vnet()

    model_path1 = os.path.join(f"./model/HAP/pancreas_{FLAGS.exp}_{FLAGS.labelnum}_labeled/self_train", 'best_model.pth')
    # model_path2 = os.path.join(f"./model/HAP/pancreas_{FLAGS.exp}_{FLAGS.labelnum}_labeled/self_train", 'best_model_v.pth')
    # model_path3 = os.path.join(f"./model/HAP/pancreas_{FLAGS.exp}_{FLAGS.labelnum}_labeled/self_train", 'best_model_r.pth')

    net1.load_state_dict(torch.load(str(model_path1)))
    # ema_net1.load_state_dict(torch.load(str(model_path2)))
    # ema_net2.load_state_dict(torch.load(str(model_path3)))

    net1.eval()
    # ema_net1.eval()
    # ema_net2.eval()

    avg_metric1 = test_all_case(net1, image_list, num_classes=num_classes,
                                patch_size=(96, 96, 96), stride_xy=16, stride_z=16,
                                save_result=False, test_save_path=test_save_path,
                                metric_detail=FLAGS.detail, nms=FLAGS.nms)

    # avg_metric2 = test_all_case_average(ema_net1, ema_net2, image_list, num_classes=num_classes,
    #                                     patch_size=(96, 96, 96), stride_xy=16, stride_z=16,
    #                                     save_result=False, test_save_path=test_save_path,
    #                                     metric_detail=FLAGS.detail, nms=FLAGS.nms)

    print("result")
    print(avg_metric1)




if __name__ == '__main__':
    testLA()

