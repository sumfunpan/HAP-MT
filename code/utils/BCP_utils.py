from locale import normalize
from multiprocessing import reduction
import pdb
from turtle import pd
import numpy as np
import torch.nn as nn
import torch
import random
from utils.losses import mask_DiceLoss
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg

DICE = mask_DiceLoss(nclass=2)
CE = nn.CrossEntropyLoss(reduction='none')



def context_mask(img, mask_ratio):
    batch_size, channel, img_x, img_y, img_z = img.shape[0],img.shape[1],img.shape[2],img.shape[3],img.shape[4]
    loss_mask = torch.ones(batch_size, img_x, img_y, img_z).cuda()
    mask = torch.ones(img_x, img_y, img_z).cuda()
    patch_pixel_x, patch_pixel_y, patch_pixel_z = int(img_x*mask_ratio), int(img_y*mask_ratio), int(img_z*mask_ratio)
    w = np.random.randint(0, img_x - patch_pixel_x)
    h = np.random.randint(0, img_y - patch_pixel_y)
    z = np.random.randint(0, img_z - patch_pixel_z)
    mask[w:w+patch_pixel_x, h:h+patch_pixel_y, z:z+patch_pixel_z] = 0
    loss_mask[:, w:w+patch_pixel_x, h:h+patch_pixel_y, z:z+patch_pixel_z] = 0
    return mask.long(), loss_mask.long()

def context_mask2(img, label, mask_ratio, ratio=1.0):
    batch_size, channel, img_x, img_y, img_z = img.shape[0],img.shape[1],img.shape[2],img.shape[3],img.shape[4]
    loss_mask = torch.ones(batch_size//2, img_x, img_y, img_z).cuda()#前半batch图像要cutmix的部分（为0表示要复制粘贴的对应区域）
    if random.random() <= ratio:
        patch_pixel_x, patch_pixel_y, patch_pixel_z = int(img_x*mask_ratio), int(img_y*mask_ratio), int(img_z*mask_ratio)
        w = np.random.randint(0, img_x - patch_pixel_x)
        h = np.random.randint(0, img_y - patch_pixel_y)
        z = np.random.randint(0, img_z - patch_pixel_z)
        loss_mask[:, w:w+patch_pixel_x, h:h+patch_pixel_y, z:z+patch_pixel_z] = 0
        mix_img1 = img[:batch_size//2, :, :, :, :].clone()
        mix_img2 = img[batch_size//2:, :, :, :, :].clone()
        mix_label1 = label[:batch_size//2, :, :, :].clone()
        mix_label2 = label[batch_size//2:, :, :, :].clone()
        mix_img1[:,0][loss_mask == 0] = img[batch_size//2:, 0, :, :, :][loss_mask == 0]
        mix_img2[:,0][loss_mask == 0] = img[:batch_size//2, 0, :, :, :][loss_mask == 0]
        mix_label1[loss_mask == 0] = label[batch_size//2:, :, :, :][loss_mask == 0]
        mix_label2[loss_mask == 0] = label[:batch_size//2, :, :, :][loss_mask == 0]
        mix_img = torch.cat((mix_img1, mix_img2), dim=0)
        mix_label = torch.cat((mix_label1, mix_label2), dim=0)
    else:
        mix_img = img
        mix_label = label
    # assert not torch.isnan(mix_img).any(), "mix_img contains NaN"
    # assert not torch.isnan(mix_label).any(), "mix_label contains NaN"

    return mix_img, mix_label, loss_mask.bool()

def context_mask3(img_a, lab_a, img_b, lab_b, ratio=0.5):#从img_b找前景贴到img_a对应位置处
    batch_size, channel, img_x, img_y, img_z = img_a.shape[0],img_a.shape[1],img_a.shape[2],img_a.shape[3],img_a.shape[4]
    apply_cutmix = torch.rand(batch_size) < ratio #对其中一部分实施复制粘贴
    # print(apply_cutmix)
    mix_img1 = img_a.clone()  #背景
    mix_img2 = img_b.clone()  #前景
    mix_label1 = lab_a.clone() #背景对应标签
    mix_label2 = lab_b.clone()
    loss_mask = torch.ones(batch_size, img_x, img_y, img_z).cuda()
    for i in range(batch_size):
        if apply_cutmix[i]:
            mask = (lab_b[i] == 1)
            loss_mask[i][mask] = 0
            mix_img1[i,0][mask] = img_b[i,0][mask]
            mix_label1[i][mask] = 1
            mix_img2[i,0][mask] = img_a[i,0][mask]
            mix_label2[i][mask] = lab_a[i][mask]


    # mask = (lab_b[apply_cutmix] == 1)
    # loss_mask = torch.ones(batch_size, img_x, img_y, img_z).cuda()
    # loss_mask[apply_cutmix][mask] = 0
    # mix_img1[apply_cutmix,0][mask] = img_b[apply_cutmix,0][mask]
    # # print(mix_img1[apply_cutmix,0].shape)
    # mix_label1[apply_cutmix][mask] = 1
    # mix_img2[apply_cutmix,0][mask] = img_a[apply_cutmix,0][mask]
    # mix_label2[apply_cutmix][mask] = lab_a[apply_cutmix][mask]
    return mix_img1, mix_label1, mix_img2, mix_label2, loss_mask.bool()



def random_mask(img):
    batch_size, channel, img_x, img_y, img_z = img.shape[0],img.shape[1],img.shape[2],img.shape[3],img.shape[4]
    loss_mask = torch.ones(batch_size, img_x, img_y, img_z).cuda()
    mask = torch.ones(img_x, img_y, img_z).cuda()
    patch_pixel_x, patch_pixel_y, patch_pixel_z = int(img_x*2/3), int(img_y*2/3), int(img_z*2/3)
    mask_num = 27
    mask_size_x, mask_size_y, mask_size_z = int(patch_pixel_x/3)+1, int(patch_pixel_y/3)+1, int(patch_pixel_z/3)
    size_x, size_y, size_z = int(img_x/3), int(img_y/3), int(img_z/3)
    for xs in range(3):
        for ys in range(3):
            for zs in range(3):
                w = np.random.randint(xs*size_x, (xs+1)*size_x - mask_size_x - 1)
                h = np.random.randint(ys*size_y, (ys+1)*size_y - mask_size_y - 1)
                z = np.random.randint(zs*size_z, (zs+1)*size_z - mask_size_z - 1)
                mask[w:w+mask_size_x, h:h+mask_size_y, z:z+mask_size_z] = 0
                loss_mask[:, w:w+mask_size_x, h:h+mask_size_y, z:z+mask_size_z] = 0
    return mask.long(), loss_mask.long()

def concate_mask(img):
    batch_size, channel, img_x, img_y, img_z = img.shape[0],img.shape[1],img.shape[2],img.shape[3],img.shape[4]
    loss_mask = torch.ones(batch_size, img_x, img_y, img_z).cuda()
    mask = torch.ones(img_x, img_y, img_z).cuda()
    z_length = int(img_z * 8 / 27)
    z = np.random.randint(0, img_z - z_length -1)
    mask[:, :, z:z+z_length] = 0
    loss_mask[:, :, :, z:z+z_length] = 0
    return mask.long(), loss_mask.long()

def mix_loss(net3_output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    dice_loss = DICE(net3_output, img_l, mask) * image_weight 
    dice_loss += DICE(net3_output, patch_l, patch_mask) * patch_weight
    loss_ce = image_weight * (CE(net3_output, img_l) * mask).sum() / (mask.sum() + 1e-16) 
    loss_ce += patch_weight * (CE(net3_output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)
    loss = (dice_loss + loss_ce) / 2
    return loss

def sup_loss(output, label):
    label = label.type(torch.int64)
    dice_loss = DICE(output, label)
    loss_ce = torch.mean(CE(output, label))
    loss = (dice_loss + loss_ce) / 2
    return loss

@torch.no_grad()
def update_ema_variables(model, ema_model, alpha):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_((1 - alpha) * param.data)

@torch.no_grad()
def update_ema_students(model1, model2, ema_model, alpha):
    for ema_param, param1, param2 in zip(ema_model.parameters(), model1.parameters(), model2.parameters()):
        ema_param.data.mul_(alpha).add_(((1 - alpha)/2) * param1.data).add_(((1 - alpha)/2) * param2.data)

@torch.no_grad()
def parameter_sharing(model, ema_model):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data = param.data

class BBoxException(Exception):
    pass

def get_non_empty_min_max_idx_along_axis(mask, axis):
    """
    Get non zero min and max index along given axis.
    :param mask:
    :param axis:
    :return:
    """
    if isinstance(mask, torch.Tensor):
        # pytorch is the axis you want to get
        nonzero_idx = (mask != 0).nonzero()
        if len(nonzero_idx) == 0:
            min = max = 0
        else:
            max = nonzero_idx[:, axis].max()
            min = nonzero_idx[:, axis].min()
    elif isinstance(mask, np.ndarray):
        nonzero_idx = (mask != 0).nonzero()
        if len(nonzero_idx[axis]) == 0:
            min = max = 0
        else:
            max = nonzero_idx[axis].max()
            min = nonzero_idx[axis].min()
    else:
        raise BBoxException("Wrong type")
    max += 1
    return min, max


def get_bbox_3d(mask):
    """ Input : [D, H, W] , output : ((min_x, max_x), (min_y, max_y), (min_z, max_z))
    Return non zero value's min and max index for a mask
    If no value exists, an array of all zero returns
    :param mask:  numpy of [D, H, W]
    :return:
    """
    assert len(mask.shape) == 3
    min_z, max_z = get_non_empty_min_max_idx_along_axis(mask, 2)
    min_y, max_y = get_non_empty_min_max_idx_along_axis(mask, 1)
    min_x, max_x = get_non_empty_min_max_idx_along_axis(mask, 0)

    return np.array(((min_x, max_x),
                     (min_y, max_y),
                     (min_z, max_z)))

def get_bbox_mask(mask):
    batch_szie, x_dim, y_dim, z_dim = mask.shape[0], mask.shape[1], mask.shape[2], mask.shape[3]
    mix_mask = torch.ones(batch_szie, 1, x_dim, y_dim, z_dim).cuda()
    for i in range(batch_szie):
        curr_mask = mask[i, ...].squeeze()
        (min_x, max_x), (min_y, max_y), (min_z, max_z) = get_bbox_3d(curr_mask)
        mix_mask[i, :, min_x:max_x, min_y:max_y, min_z:max_z] = 0
    return mix_mask.long()

