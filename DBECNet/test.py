import sys

from matplotlib import pyplot as plt

sys.path.insert(0, '.')

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn.parallel import gather
import torch.optim.lr_scheduler

import dataset as myDataLoader
import Transforms as myTransforms
from metric_tool import ConfuseMatrixMeter
from PIL import Image

import os, time
import numpy as np
from argparse import ArgumentParser
from models.model import  HSSNet


def ValidateSegmentation(args):
    torch.backends.cudnn.benchmark = True
    SEED = 2333
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    model = HSSNet(3, 1)

    if args.file_root == 'LEVIR-cd256':
        args.file_root = '/home/ww/sq/data/LEVIR-cd256'
    elif args.file_root == 'WHU':
        args.file_root = '/home/ww/sq/data/WHU'
    elif args.file_root == 'SYSU':
        args.file_root = '/home/ww/sq/data/SYSU'
    elif args.file_root == 'CDD':
        args.file_root = '/home/ww/sq/data/CDD'
    else:
        raise TypeError('%s has not defined' % args.file_root)

    if not os.path.exists(args.vis_dir):
        os.makedirs(args.vis_dir)

    if args.onGPU:
        model = model.cuda()

    mean = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
    std = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]

    # compose the data with transforms
    valDataset = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.ToTensor()
    ])

    test_data = myDataLoader.Dataset("test", file_root=args.file_root, transform=valDataset)
    testLoader = torch.utils.data.DataLoader(
        test_data, shuffle=False,
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=False)

    if args.onGPU:
        cudnn.benchmark = True

    # load the model
    model_file_name = args.weight
    state_dict = torch.load(model_file_name)
    model.load_state_dict(state_dict,strict=False)
    model.eval()

    with torch.no_grad():
        salEvalVal = ConfuseMatrixMeter(n_class=2)

        start = time.time()
        for iter, batched_inputs in enumerate(testLoader):
            img, target = batched_inputs
            img_name = testLoader.sampler.data_source.file_list[iter]
            pre_img = img[:, 0:3]
            post_img = img[:, 3:6]

            if args.onGPU == True:
                pre_img = pre_img.cuda()
                target = target.cuda()
                post_img = post_img.cuda()

            pre_img_var = torch.autograd.Variable(pre_img).float()
            post_img_var = torch.autograd.Variable(post_img).float()
            target_var = torch.autograd.Variable(target).float()

            # run the mdoel
            output = model(pre_img_var, post_img_var)
            pred = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output)).long()

            # save change maps
            pr = pred[0, 0].cpu().numpy()
            gt = target_var[0, 0].cpu().numpy()

            index_tp = np.where(np.logical_and(pr == 1, gt == 1))
            index_fp = np.where(np.logical_and(pr == 1, gt == 0))
            index_tn = np.where(np.logical_and(pr == 0, gt == 0))
            index_fn = np.where(np.logical_and(pr == 0, gt == 1))

            map = np.zeros([gt.shape[0], gt.shape[1], 3])
            #tp,fp,tn,fn
            map[index_tp] = [255, 255, 255]
            map[index_fp] = [255, 0, 0]
            map[index_tn] = [0, 0, 0]
            map[index_fn] = [0, 255, 255]
            change_map = Image.fromarray(np.array(map, dtype=np.uint8))
            save_path = os.path.join(args.vis_dir, img_name)
            change_map.save(save_path)

            f1 = salEvalVal.update_cm(pr, gt)

            print('{}/{}, {}'.format(iter+1, len(testLoader), img_name))
        print(time.time()-start)
        scores = salEvalVal.get_scores()
        print('Precision:{:.4f}, Recall:{:.4f}, F1:{:.4f}, IoU:{:.4f}, OA:{:.4f}'.format(scores['precision'], scores['recall'], scores['F1'], scores['Iou'], scores['OA']))

    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--file_root', default="LEVIR-cd256", help='Data directory | LEVIR | WHU | SYSU | CDD ')
    parser.add_argument('--inWidth', type=int, default=256, help='Width of RGB image')
    parser.add_argument('--inHeight', type=int, default=256, help='Height of RGB image')
    parser.add_argument('--num_workers', type=int, default=4, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--vis_dir', default='/home/ww/sq/HSS1/LEVIR_VIS', help='Directory to save the results')
    parser.add_argument('--onGPU', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Run on CPU or GPU. If TRUE, then GPU.')
    parser.add_argument('--weight', default='/home/ww/sq/HSS1/result/LEVIR.log/LEVIR-cd256_iter_68200_lr_0.0001/Epoch80_0.9123454168784871.pth', type=str, help='pretrained weight, can be a non-strict copy')
    parser.add_argument('--ms', type=int, default=0, help='apply multi-scale training, default False')
    args = parser.parse_args()
    print('Called with args:')
    print(args)

    ValidateSegmentation(args)
