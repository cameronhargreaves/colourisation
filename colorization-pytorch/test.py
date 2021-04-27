import os
from options.train_options import TrainOptions
from models import create_model
from util.visualizer import save_images
from util import html
import math
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import string
import torch
import torchvision
import torchvision.transforms as transforms

import datetime as dt
import matplotlib.pyplot as plt

from util import util
import numpy as np

if __name__ == '__main__':
    # sample_ps = [1., .125, .03125]
    # to_visualize = ['gray', 'hint', 'hint_ab', 'fake_entr', 'real', 'fake_reg', 'real_ab', 'fake_ab_reg', ]
    
    to_visualize = ['gray', 'real', 'fake_reg']
    sample_ps = [0.03125]

    # num_points = np.round(10**np.arange(-.1, 2.8, .1))
    # num_points[0] = 0
    # num_points = np.unique(num_points.astype('int'))
    # N = len(num_points)

    S = len(sample_ps)

    opt = TrainOptions().parse()
    opt.load_model = True
    opt.num_threads = 1  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.display_id = -1  # no visdom display
    opt.phase = 'val'
    # opt.dataroot = '/data/cifar10png/test'
    opt.serial_batches = True
    opt.aspect_ratio = 1.
    opt.how_many = 56000

    testset = torchvision.datasets.ImageFolder(root='/home/cam/Desktop/datasets/google/', transform=transforms.Compose([
            transforms.Resize((opt.loadSize, opt.loadSize)),
            transforms.ToTensor()]))

    # testset = torchvision.datasets.CIFAR10(root='./data', train=True,
    #                                        download=True, transform=transforms.Compose([
    #         transforms.Resize((opt.loadSize, opt.loadSize)),
    #         transforms.ToTensor()]))

    # testset = torchvision.datasets.ImageFolder(opt.dataroot,
    #                                            transform=transforms.Compose([
    #                                                transforms.Resize((opt.loadSize, opt.loadSize)),
    #                                                transforms.ToTensor()]))

    # testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,transform=transforms.Compose([
    #                                                transforms.Resize((opt.loadSize, opt.loadSize)),
    #                                                transforms.ToTensor()]))

    testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size, shuffle=False,
                                             num_workers=int(opt.num_threads))

    model = create_model(opt)
    model.setup(opt)
    model.eval()

    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

    # statistics
    psnrs = np.zeros(opt.how_many)
    mses = np.zeros(opt.how_many)
    maes = np.zeros(opt.how_many)
    min_mae = 10000000
    min_mse = 10000000
    min_psnr = 10000000
    max_mae = 0
    max_mse = 0
    max_psnr = 0

    # psnrs = np.zeros((opt.how_many, N))

    entrs = np.zeros((opt.how_many, S))
    f = open("google_open_images_values.txt", "a")

    for i, data_raw in enumerate(testloader):
        try :
            data_raw[0] = data_raw[0].cuda()
            # data_raw[0] = util.crop_mult(data_raw[0], mult=8)

            # with no points
            # for nn in range(N):
            for (pp, sample_p) in enumerate(sample_ps):
                img_path = [str.replace('%08d_%.3f' % (i, sample_p), '.', 'p')]
                data = util.get_colorization_data(data_raw, opt, ab_thresh=0., p=sample_p)
                # data = util.get_colorization_data(data_raw, opt, ab_thresh=0., num_points=num_points[nn])

                model.set_input(data)
                model.test(True)  # True means that losses will be computed
                visuals = util.get_subset_dict(model.get_current_visuals(), to_visualize)
                psnrs[i] = util.calculate_psnr_np(util.tensor2im(visuals['real']), util.tensor2im(visuals['fake_reg']))
                mses[i] = util.calculate_mse(util.tensor2im(visuals['real']), util.tensor2im(visuals['fake_reg']))
                maes[i] =util.calculate_mae(util.tensor2im(visuals['real']),util.tensor2im(visuals['fake_reg']))

                if (psnrs[i] > max_psnr) :
                    max_psnr = psnrs[i]

                if (psnrs[i] < min_psnr) :
                    min_psnr = psnrs[i]

                if (mses[i] < min_mse):
                    min_mse = mses[i]

                if (mses[i] > max_mse):
                    max_mse = mses[i]

                if (maes[i] < min_mae):
                    min_mae = maes[i]

                if (maes[i] > max_mae) :
                    max_mae = maes[i]

                # entrs[i, pp] = model.get_current_losses()['G_entr']

                # save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize)
        except :
            continue

        if i % 100 == 0:
            print('.......................processing data - ' + str(math.trunc((i / opt.how_many) * 100)) + '% .......................', end="\r")
            # print('processing (%04d)-th image... %s' % (i, img_path))

        print('.......................processing file - ' + str(i) +' out of ' + str(opt.how_many) + ' ....................... PSNR - ' + str(psnrs[i]), end="\r")
        f.write('\nImage ' + str(i) + ' |  PSNR - ' + str(psnrs[i]) + ' | MSE - ' + str(mses[i]) + ' | MAE - ' + str(maes[i]))

        if i == opt.how_many - 1:
            break


    f.close()
    # Compute and print some summary statistics
    psnrs_mean = np.mean(psnrs)
    maes_mean = np.mean(maes)
    mses_mean = np.mean(mses)

    psnrs_std = np.std(psnrs, axis=0) / np.sqrt(opt.how_many)
    maes_std = np.std(maes, axis=0) / np.sqrt(opt.how_many)
    mses_std = np.std(mses, axis=0) / np.sqrt(opt.how_many)

    print('\nPSNR Mean : ' + str(psnrs_mean))
    print('\nPSNR Max - ' + str(max_psnr))
    print('\nPSNR Min - ' + str(min_psnr))
    print('\nPSNR Standard Deviation - ' + str(psnrs_std))

    print('\nMAE Mean : ' + str(maes_mean))
    print('\nMAE Min- ' + str(min_mae))
    print('\nMAE Max - ' + str(max_mae))
    print('\nMAE Standard Deviation - ' + str(maes_std))

    print('\nMSE Mean : ' + str(mses_mean))
    print('\nMinimum MSE- ' + str(min_mse))
    print('\nMaximum MSE- ' + str(max_mse))
    print('\nMSE Standard Deviation - ' + str(mses_std))
