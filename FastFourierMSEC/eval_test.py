'''
Test Vid4 (SR) and REDS4 (SR-clean, SR-blur, deblur-clean, deblur-compression) datasets
'''
import os
import os.path as osp
import glob
import logging
import numpy as np
import cv2
import torch
import argparse

import utils.util as util
import data.util as data_util
import models.archs.EDVR_arch as EDVR_arch


def main(root, output, videoLists, modelPath):
    #################
    # configurations
    #################
    device = torch.device('cuda')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    data_mode = 'Vid4'  # Vid4 | sharp_bicubic | blur_bicubic | blur | blur_comp
    # Vid4: SR
    # REDS4: sharp_bicubic (SR-clean), blur_bicubic (SR-blur);
    #        blur (deblur-clean), blur_comp (deblur-compression).
    flip_test = False
    ############################################################################
    #### model

    model_path =  modelPath
    # model_path = '/output/experiments/EDVR_NEW/models/40_G.pth'
    N_in = 5
    predeblur, HR_in = False, False
    back_RBs = 10

    model = EDVR_arch.EDVR(nf=32, nframes=N_in, groups=8, front_RBs=5, back_RBs=back_RBs, predeblur=predeblur,
                           HR_in=HR_in)

    #### dataset
    test_dataset_folder = root

    #### evaluation
    # temporal padding mode

    padding = 'new_info'
    save_imgs = True

    save_folder = output
    util.mkdirs(save_folder)
    util.setup_logger('base', save_folder, 'test', level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')

    #### log info
    logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
    logger.info('Padding mode: {}'.format(padding))
    logger.info('Model path: {}'.format(model_path))
    logger.info('Save images: {}'.format(save_imgs))
    logger.info('Flip test: {}'.format(flip_test))

    #### set up the models
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    model = model.to(device)

    subfolder_name_l = []

    with open(videoLists, 'r') as f:
        while True:
            line = f.readline().strip()
            if line == '':
                break
            subfolder_name_l.append(osp.join(root, line))

    subfolder_l = sorted(subfolder_name_l)
    subfolder_name_l = []

    print(subfolder_l)
    # temp = input()

    # subfolder_l = sorted(glob.glob(osp.join(test_dataset_folder, '*')))
    # for each subfolder
    i = 0
    for subfolder in subfolder_l:
        subfolder_name = osp.basename(subfolder)
        subfolder_name_l.append(subfolder_name)
        save_subfolder = osp.join(save_folder, subfolder_name)

        img_path_l = sorted(glob.glob(osp.join(subfolder, '*')))
        max_idx = len(img_path_l)
        if save_imgs:
            util.mkdirs(save_subfolder)

        #### read LQ and GT images
        imgs_LQ = data_util.read_img_seq(subfolder)
        i = i + 1
        print("process %d th subfolder" % i)

        # process each image
        j = 0
        for img_idx, img_path in enumerate(img_path_l):
            img_name = osp.splitext(osp.basename(img_path))[0]
            select_idx = data_util.index_generation(img_idx, max_idx, N_in, padding=padding)
            imgs_in = (imgs_LQ.index_select(0, torch.LongTensor(select_idx))[2]).unsqueeze(0).to(device)

            if flip_test:
                output = util.flipx4_forward(model, imgs_in)
            else:
                output = util.single_forward(model, imgs_in)
            output = util.tensor2img(output.squeeze(0))

            if save_imgs:
                cv2.imwrite(osp.join(save_subfolder, '{}.png'.format(img_name)), output)

            j = j + 1
            print("process %d th image" % j)

    logger.info('################ Finish Testing ################')
    # logger.info('Data: {} - {}'.format(data_mode, test_dataset_folder))
    # logger.info('Padding mode: {}'.format(padding))
    # logger.info('Model path: {}'.format(model_path))
    # logger.info('Save images: {}'.format(save_imgs))
    # logger.info('Flip test: {}'.format(flip_test))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/tmp/data/answer/stage1_test_input',  metavar='PATH', help='validation dataset root dir')
    parser.add_argument('--videoLists', type=str, default='/code/code_enhance/data/videolist_test1.txt', metavar='FILE', help='record video ids')
    parser.add_argument('--output', type=str, default='/tmp/data/answer/stage2_test_input/', metavar='PATH', help='save results')
    parser.add_argument('--modelPath', type=str, default='/tmp/data/model/stage1_model/experiments/EDVR_NEW/models/140000_G.pth', help='Model path')

    args = parser.parse_args()

    root = args.root
    videoLists = args.videoLists
    output = args.output
    modelPath = args.modelPath

    main(root, output, videoLists, modelPath)
