import os
import argparse
import torch
import torch.nn as nn
from Model.SAT_Pro2SAM import *
from DataLoader import *
from torch.autograd import Variable
from utils.accuracy import *
from utils.lr import *
from utils.util import copy_dir, makedirs
from utils.optimizer import *
import random
from skimage import measure
import cv2
import datetime
import time
from utils.func import *
from evaluator import val_loc_one_epoch
import sys
import pprint
import collections
import shutil
from utils.optimizer import create_optimizerv2
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from skimage import data,filters,segmentation,measure,morphology,color

from GroupMixFormer.models.groupmixformer import GroupMixFormer
from GroupMixFormer.models.Pro2Mask import Pro2Mask, ClassificationNet
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

seed = 6
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def trans_bool_tensor(mask, temp=1):
    new_mask = np.zeros(mask.shape)
    new_mask[mask == True] = temp
    return new_mask

def calculate_box_iou(mask1, mask2):
    mask1_coor = multiboxes_from_mask(mask1)
    box1 = mask1_coor
    mask2_coor = multiboxes_from_mask(mask2)
    box2=mask2_coor

    # 计算两个框的交集的左上角和右下角坐标
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # 计算交集的面积
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # 计算两个框的面积
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # 计算IOU值
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou

def filter_tensor(mask, temp=0.2):
    # new_mask = np.zeros(mask.shape)
    new_tensor = torch.zeros(mask.shape)
    new_tensor[mask >temp] = 1
    return new_tensor

def calculate_iou(mask1, mask2):
    intersection = torch.logical_and(mask1, mask2).sum().float()
    union = torch.logical_or(mask1, mask2).sum().float()
    iou = intersection / union

    return iou

def val_loc_one_epoch(args, Val_Loader, model, epoch=0):
    thr_num = len(args.threshold)
    cls_top1 = AverageMeter()
    cls_top5 = AverageMeter()
    IoU30 = [AverageMeter() for i in range(thr_num)]
    IoU50 = [AverageMeter() for i in range(thr_num)]
    IoU70 = [AverageMeter() for i in range(thr_num)]
    loc_gt = [AverageMeter() for i in range(thr_num)]
    loc_top1 = [AverageMeter() for i in range(thr_num)]
    loc_top5 = [AverageMeter() for i in range(thr_num)]
    best_thr_num = 0
    best_loc = 0
    with torch.no_grad():
        model.eval()
        for step, (img, img_tencrop, label, gt_boxes, path) in enumerate(Val_Loader):
            if args.evaluate == False:
                return None, None, None
            img, img_tencrop, label = Variable(img).cuda(), Variable(img_tencrop).cuda(), label.cuda()
            output1 = model(img)
            map = torch.zeros([256, 14, 14])
            map = map.data.cpu()
            map = np.array(map.data.cpu())
            batch = label.size(0)
            for i in range(batch):
                map_i = map[i]
                map_i = normalize_map(map_i, args.crop_size)
                gt_boxes_i = gt_boxes[i]
                if args.root == 'Stanford_Dogs' or args.root == 'ILSVRC':
                    gt_boxes_i = gt_boxes[i].strip().split(' ')
                    gt_boxes_i = np.array([float(gt_boxes_i[xxx]) for xxx in range(len(gt_boxes_i))])
                label_i = label[i].unsqueeze(0)
                output1_i = output1[i].unsqueeze(0)
                gt_boxes_i = np.reshape(gt_boxes_i, -1)
                gt_box_num = len(gt_boxes_i) // 4
                gt_boxes_i = np.reshape(gt_boxes_i, (gt_box_num, 4))

                ##  tencrop_cls
                if args.tencrop:
                    output1_i = model(img_tencrop[i])
                    output1_i = nn.Softmax()(output1_i)
                    output1_i = torch.mean(output1_i, dim=0, keepdim=True)
                prec1, prec5 = accuracy(output1_i.data, label_i, topk=(1, 5))
                cls_top1.updata(prec1, 1)
                cls_top5.updata(prec5, 1)
                ##  loc_acc
                for j in range(thr_num):
                    highlight = np.zeros(map_i.shape)
                    highlight[map_i > args.threshold[j]] = 1
                    all_labels = measure.label(highlight)
                    highlight = np.zeros(highlight.shape)
                    highlight[all_labels == count_max(all_labels.tolist())] = 1
                    highlight = np.round(highlight * 255)
                    highlight_big = cv2.resize(highlight, (args.crop_size, args.crop_size),
                                               interpolation=cv2.INTER_NEAREST)
                    props = measure.regionprops(highlight_big.astype(int))
                    best_bbox = [0, 0, args.crop_size, args.crop_size]
                    if len(props) == 0:
                        bbox = [0, 0, args.crop_size, args.crop_size]
                    else:
                        temp = props[0]['bbox']
                        bbox = [temp[1], temp[0], temp[3], temp[2]]

                    max_iou = -1
                    for m in range(gt_box_num):
                        iou = IoU(bbox, gt_boxes_i[m])
                        if iou > max_iou:
                            max_iou = iou
                            max_box_num = m
                            best_bbox = bbox
                    ##  gt_loc
                    loc_gt[j].updata(100, 1) if max_iou >= 0.5 else loc_gt[j].updata(0, 1)
                    ##  maxboxaccv2
                    IoU30[j].updata(100, 1) if max_iou >= 0.3 else IoU30[j].updata(0, 1)
                    IoU50[j].updata(100, 1) if max_iou >= 0.5 else IoU50[j].updata(0, 1)
                    IoU70[j].updata(100, 1) if max_iou >= 0.7 else IoU70[j].updata(0, 1)
                    cls_loc = 100 if prec1 and max_iou >= 0.5 else 0
                    cls_loc_5 = 100 if prec5 and max_iou >= 0.5 else 0
                    loc_top1[j].updata(cls_loc, 1)
                    loc_top5[j].updata(cls_loc_5, 1)

                ## save_img
                if args.save_img_flag and ((args.root != 'ILSVRC') or (args.root == 'ILSVRC' and step < 1)):
                    new_path = path[i].replace('\\', '/')
                    ori_img = cv2.imread(new_path)
                    ori_img = cv2.resize(ori_img, (args.resize_size, args.resize_size))
                    shift = (args.resize_size - args.crop_size) // 2
                    if shift > 0:
                        ori_img = ori_img[shift:-shift, shift:-shift, :]
                    heatmap = np.uint8(255 * map_i)
                    img_name = new_path.split('/')[-1]
                    heatmap = heatmap.astype(np.uint8)
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                    img_add = cv2.addWeighted(ori_img.astype(np.uint8), 0.5, heatmap.astype(np.uint8), 0.5, 0)
                    if args.save_box_flag:
                        cv2.rectangle(img_add, (int(best_bbox[0]), int(best_bbox[1])),
                                      (int(best_bbox[2]), int(best_bbox[3])), (0, 255, 0), 4)
                        cv2.rectangle(img_add, (int(gt_boxes_i[max_box_num][0]), int(gt_boxes_i[max_box_num][1])),
                                      (int(gt_boxes_i[max_box_num][2]), int(gt_boxes_i[max_box_num][3])), (0, 0, 255),
                                      4)
                    if os.path.exists(
                            args.save_path + '/' + args.root + '/' + args.arch + '/' + args.save_img_dir) == 0:
                        os.mkdir(args.save_path + '/' + args.root + '/' + args.arch + '/' + args.save_img_dir)
                    cv2.imwrite(
                        args.save_path + '/' + args.root + '/' + args.arch + '/' + args.save_img_dir + '/' + img_name,
                        img_add)

            if args.root == 'ILSVRC' and (step + 1) % (len(Val_Loader) // 10) == 0:
                best_loc = 0
                for j in range(thr_num):
                    if (loc_gt[j].avg + loc_top1[j].avg) > best_loc:
                        best_thr_num = j
                        best_loc = loc_gt[j].avg + loc_top1[j].avg
                print(
                    'step:[{}/{}]\t thr: {:.2f}  \t gt_loc : {:.2f}  \t loc_top1 : {:.2f} \t loc_top5 : {:.2f} '.format(
                        step + 1, len(Val_Loader), args.threshold[best_thr_num], loc_gt[best_thr_num].avg,
                        loc_top1[best_thr_num].avg, loc_top5[best_thr_num].avg))
        print('Val Epoch : [{}][{}/{}]  \n'
              'cls_top1 : {:.2f} \t cls_top5 : {:.2f} '.format(epoch, step + 1, len(Val_Loader), cls_top1.avg,
                                                               cls_top5.avg))
        best_loc = 0
        for j in range(thr_num):
            if (loc_gt[j].avg + loc_top1[j].avg) > best_loc:
                best_thr_num = j
                best_loc = loc_gt[j].avg + loc_top1[j].avg
        print('thr: {:.2f}  \t gt_loc : {:.2f}  \t loc_top1 : {:.2f} \t loc_top5 : {:.2f} \t '.format(
            args.threshold[best_thr_num], loc_gt[best_thr_num].avg, loc_top1[best_thr_num].avg,
            loc_top5[best_thr_num].avg))
        print('IoU30: {:.2f}  \t IoU50 : {:.2f}  \t IoU70 : {:.2f} \t Mean : {:.2f}  '.format(IoU30[best_thr_num].avg,
                                                                                              IoU50[best_thr_num].avg,
                                                                                              IoU70[best_thr_num].avg, (
                                                                                                          IoU30[
                                                                                                              best_thr_num].avg +
                                                                                                          IoU50[
                                                                                                              best_thr_num].avg +
                                                                                                          IoU70[
                                                                                                              best_thr_num].avg) / 3))
    return loc_top1[best_thr_num].avg, loc_gt[best_thr_num].avg, args.threshold[best_thr_num]


def multiboxes_from_mask(mask, thr=10):
    image = mask
    image = image.data.cpu()
    image = np.array(image.data.cpu())
    thresh = filters.threshold_otsu(image)  # 阈值分割

    bw = morphology.closing(image > thresh, morphology.square(3))  # 闭运算

    cleared = bw.copy()  # 复制
    segmentation.clear_border(cleared)  # 清除与边界相连的目标物

    label_image = measure.label(cleared)  # 连通区域标记
    borders = np.logical_xor(bw, cleared)  # 异或
    label_image[borders] = -1
    # image_label_overlay = color.label2rgb(label_image, image=image)  # 不同标记用不同颜色显示

    boxes_list = []
    coor_list = []

    x_list = []
    y_list = []
    if len(measure.regionprops(label_image)) == 0:
        bbox = [0, 0, 224, 224]
    else:
        for region in measure.regionprops(label_image):  # 循环得到每一个连通区域属性集
            # 忽略小区域
            if region.area < thr:
                continue
            # 绘制外包矩形
            minr, minc, maxr, maxc = region.bbox
            y, x = region.centroid
            x, y = int(x), int(y)

            # box = [minc, minr, maxc, maxr, x, y]  # x1, y1, x2, y2, x_center, y_center
            box = [minc, minr, maxc, maxr]  # x1, y1, x2, y2
            coor = [x, y]
            boxes_list.append(box)
            coor_list.append(coor)
            x_list.append(minc)
            x_list.append(maxc)
            y_list.append(minr)
            y_list.append(maxr)
        if len(x_list) ==0 and len(y_list) ==0:
            bbox = [0, 0, 224, 224]
        else:
            x_min = min(x_list)
            x_max = max(x_list)

            y_min = min(y_list)
            y_max = max(y_list)
            bbox = [x_min, y_min, x_max, y_max]

    return bbox

#
# def multiboxes_from_mask(mask):
#     image = mask
#     thresh = filters.threshold_otsu(image)  # 阈值分割
#
#     bw = morphology.closing(image > thresh, morphology.square(3))  # 闭运算
#
#     cleared = bw.copy()  # 复制
#     segmentation.clear_border(cleared)  # 清除与边界相连的目标物
#
#     label_image = measure.label(cleared)  # 连通区域标记
#     borders = np.logical_xor(bw, cleared)  # 异或
#     label_image[borders] = -1
#     # image_label_overlay = color.label2rgb(label_image, image=image)  # 不同标记用不同颜色显示
#
#     boxes_list = []
#     coor_list = []
#
#     for region in measure.regionprops(label_image):  # 循环得到每一个连通区域属性集
#         # 忽略小区域
#         if region.area < 5:
#             continue
#         # 绘制外包矩形
#         minr, minc, maxr, maxc = region.bbox
#         y, x = region.centroid
#         x, y = int(x), int(y)
#
#         # box = [minc, minr, maxc, maxr, x, y]  # x1, y1, x2, y2, x_center, y_center
#         box = [minc, minr, maxc, maxr]  # x1, y1, x2, y2
#         coor = [x, y]
#         boxes_list.append(box)
#         coor_list.append(coor)
#
#     return boxes_list, coor_list

def get_mask_for_autoSAM(image, mask, mask_generator):
    # obtain all mask of a image from sam
    masks_from_sam = mask_generator.generate(image)

    iou_scores = []
    mask_all_from_saM = []

    for i, mask_sam in enumerate(masks_from_sam):
        mask_sam = mask_sam['segmentation']
        mask_sam = trans_bool_tensor(mask_sam)
        box = multiboxes_from_mask(torch.from_numpy(mask_sam))
        # print(box)
        if box == [0, 0, 224, 224]:
            print(" This mask is a error segment!")
            continue
        mask_all_from_saM.append(mask_sam)
        mask_sam = torch.from_numpy(mask_sam).cuda(device=0)

        # print(sam_mask.shape)
        h, w = mask_sam.shape
        # trans the size for keep same size
        mask_sam = F.interpolate(mask_sam.view(1, 1, h, w), size=(14, 14), mode='bilinear').squeeze().squeeze()

        # binary the mask to [0, 1], default filter value is 0.5
        mask = filter_tensor(mask).cuda(device=0)
        # computing the iou value to evaluate the mask from sam
        iou = calculate_iou(mask, mask_sam)  # two input masks keep the same type (tensor) and size

        iou_scores.append(iou)
        # mask_all.append()

    # print(iou_scores)
    id_mask = iou_scores.index(max(iou_scores))
    # print("the best mask is = {}".format(id_mask))
    pesduo_mask = mask_all_from_saM[id_mask]
    return pesduo_mask, iou_scores[id_mask]

def get_mask_from_pointSAM(image, mask, predictor):
    # computing the box and coord of foreground map
    H, W, C = image.shape
    # print("image size is {}, {}".format(H, W))
    boxex, coordx = multiboxes_from_mask(mask.cpu().numpy())

    # obtain coord of mask, x1 is col, y1 is row
    x1, y1 = coordx[0]

    # print("the trans before coord is {}, {}".format(x1, y1))
    x1 = int((x1 / 14) * W)
    y1 = int((y1 / 14) * H)
    input_coord = np.array([[x1, y1]])
    print("the trans after coord is {}, {}".format(x1, y1))

    # obtain three masks from point prompt, note that the order of input coord is [col, row]
    predictor.set_image(image)
    masks_from_sam, scores, logits = predictor.predict(
                    point_coords=input_coord,
                    point_labels=np.array([1]),
                    multimask_output=True,
                    )

    iou_scores = []

    # binary the mask to [0, 1], default filter value is 0.5
    mask = filter_tensor(mask).cuda(device=0)

    for i in range(len(masks_from_sam)):
        sam_mask = masks_from_sam[i]
        sam_mask = trans_bool_tensor(sam_mask)
        sam_mask = torch.from_numpy(sam_mask).cuda(device=0)

        h, w = sam_mask.shape
        sam_mask = F.interpolate(sam_mask.view(1, 1, h, w), size=(14, 14), mode='bilinear').squeeze().squeeze()

        iou = calculate_iou(mask, sam_mask)     # two input masks keep the same type (tensor) and size
        iou_scores.append(iou)

    id_mask = iou_scores.index(max(iou_scores))
    print("the best mask is = {}".format(id_mask))
    pesduo_mask = masks_from_sam[id_mask]
    pesduo_mask = trans_bool_tensor(pesduo_mask)

    return pesduo_mask, iou_scores[id_mask]



parser = argparse.ArgumentParser()
##  path
parser.add_argument('--root', type=str,
                    help="[CUB_200_2011, ILSVRC, OpenImage, Fgvc_aircraft_2013b, Standford_Car, Stanford_Dogs]",
                    default='CUB_200_2011')
parser.add_argument('--num_classes', type=int, default=200)
##  save
parser.add_argument('--save_path', type=str, default='logs')
parser.add_argument('--log_file', type=str, default='log.txt')
parser.add_argument('--log_code_dir', type=str, default='save_code')
##  image
parser.add_argument('--crop_size', type=int, default=224)
parser.add_argument('--resize_size', type=int, default=224)
##  dataloader
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--weight_decay', type=float, default=5e-2)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--opt', type=str, default='adamw')
##  train
parser.add_argument('--warmup_epochs', type=int, default=0)
parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N')
parser.add_argument('--weight_decay_end', type=float, default=None)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--phase', type=str, default='train')
parser.add_argument('--drop', type=float, default=0.0)
parser.add_argument('--drop_path', type=float, default=0.1)
parser.add_argument('--update_freq', default=1, type=int)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--area_thr', type=float, default=0.25)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--min_lr', type=float, default=1e-6)
##  model
parser.add_argument('--arch', type=str, default='deit_sat_small_patch16_224')

##  evaluate
parser.add_argument('--save_img_flag', type=bool, default=False)
parser.add_argument('--save_error_flag', type=bool, default=False)
parser.add_argument('--tencrop', type=bool, default=False)
parser.add_argument('--evaluate', type=bool, default=False)
parser.add_argument('--threshold', type=float, default=[0.3, 0.35, 0.4, 0.45])
parser.add_argument('--evaluate_epoch', default=1, type=int)
##  GPU'
parser.add_argument('--gpu', type=str, default='0,1,2,3')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
lr = args.lr

## save_log_txt
makedirs(args.save_path + '/' + args.root + '/' +args.arch + '/' + args.log_code_dir)
sys.stdout = Logger(args.save_path + '/' + args.root + '/' + args.arch + '/' + args.log_code_dir + '/' + args.log_file)
pprint.pprint(args)
print('\n')
sys.stdout.log.flush()

##  save_code
save_file = ['train.py', 'evaluator.py', 'evaluator_ImageNet.py']
for file_name in save_file:
    shutil.copyfile(file_name,
                    args.save_path + '/' + args.root + '/' + '/' + args.log_code_dir + '/' + file_name)
save_dir = ['Model', 'utils', 'DataLoader']
for dir_name in save_dir:
    copy_dir(dir_name, args.save_path + '/' + args.root + '/' + '/' + args.log_code_dir + '/' + dir_name)


def evaluate_epoch(root, args, Val_Loader, model, epoch):
    top1_acc, gt_acc, thr = {"CUB_200_2011": val_loc_one_epoch,
                             "ILSVRC": val_loc_one_epoch,
                             "Fgvc_aircraft_2013b": val_loc_one_epoch,
                             "Standford_Car": val_loc_one_epoch,
                             "Stanford_Dogs": val_loc_one_epoch,
                             "OpenImage": val_loc_one_epoch,
                             }[root](args, Val_Loader, model, epoch + 1)
    return top1_acc, gt_acc, thr


if __name__ == '__main__':
    ##  dataloader
    TrainData = eval(args.root).ImageDataset(args, phase='train')
    ValData = eval(args.root).ImageDataset(args, phase='test')
    Train_Loader = torch.utils.data.DataLoader(dataset=TrainData, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True)
    Val_Loader = torch.utils.data.DataLoader(dataset=ValData, batch_size=128, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)
    #  model
    model = eval(args.arch)(num_classes=args.num_classes, drop_rate=args.drop, drop_path_rate=args.drop_path,
                            pretrained=True).cuda()
    model = nn.DataParallel(model, device_ids=[int(ii) for ii in range(int(torch.cuda.device_count()))])
    model.cuda(device=0)

    checkpoint = torch.load('logs/' + args.root + '/' + args.arch + '/' + 'best_loc.pth.tar')
    model.load_state_dict(checkpoint['model'])

    sam = sam_model_registry["vit_h"](checkpoint="/media/gpu/dss/code/pre_model_pth/SAM/sam_vit_h_4b8939.pth")
    sam = sam.cuda(device=0)
    mask_generator = SamAutomaticMaskGenerator(model=sam
                                               )  # obtain all masks by a 32 * 32 grid point
    predictor = SamPredictor(sam)  # obtain three masks by a given point prompt or box prompt

    total_batch_size = args.batch_size * args.update_freq
    num_training_steps_per_epoch = len(TrainData) // total_batch_size

    root_path = '/media/gpu/dss/code/WSOL/Pro2Mask/Data/CUB_200_2011/'
    print('\nTrain begining!')
    start_time = time.time()

    for step, (path, imgs, label) in enumerate(Train_Loader):
        imgs, label = Variable(imgs).cuda(), label.cuda()
        # print(path[0])
        path = path[0]
        image_name, image_class = path.split('/')[-1], path.split('/')[-2]
        sub_root = os.path.join(root_path, 'train_sam_mask_IOU', image_class)
        if not os.path.exists(sub_root):
            os.makedirs(sub_root)
        save_path = os.path.join(sub_root, image_name)

        # model
        outputs = model(imgs)
        # get predict mask from pro2mask
        pre_mask = outputs[1].squeeze().squeeze()
        # pre_mask = filter_tensor(pre_mask)


        # pre_mask = pre_mask.data.cpu().numpy().squeeze().squeeze()
        # print(pre_mask.shape)

        # obtain ndarry type image
        image_path = path.replace('\\', '/')
        sam_image = cv2.imread(image_path)
        sam_image = cv2.cvtColor(sam_image, cv2.COLOR_BGR2RGB)
        sam_image = cv2.resize(sam_image, dsize=(224, 224))

        # obtain the pseudo mask from SAM
        pseudo_mask, iou_scores = get_mask_for_autoSAM(sam_image, pre_mask, mask_generator)
        # pre_mask = pre_mask * 255
        # pre_mask = filter_tensor(pre_mask)
        cv2.imwrite(save_path, pseudo_mask * 255)
        # print(pre_mask)
        print("the save mask: {} &&& the iou score: {}".format(save_path, iou_scores))
        # print("the save mask is : {}".format(save_path))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))



