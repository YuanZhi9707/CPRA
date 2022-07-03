import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from DerainDataset import prepare_data_Rain100L, Dataset
from utils import *
from utils import findLastCheckpoint, batch_PSNR
from torch.optim.lr_scheduler import MultiStepLR
from SSIM import *
from net import *

parser = argparse.ArgumentParser(description="Network")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=[40, 60, 80],
                    help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=0.001, help="Initial learning rate")
parser.add_argument("--save_path", type=str, default="logs/Rain100L", help='path of log files')
parser.add_argument("--save_freq", type=int, default=1, help='save intermediate model')
parser.add_argument("--data_path", type=str, default="datasets/train/RainTrainL", help='path to training data')
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
opt = parser.parse_args()

if opt.use_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id


def main():
    print('Loading dataset ...\n')
    dataset_train = Dataset(data_path=opt.data_path)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batch_size,
                              shuffle=True)
    print("# of training samples: %d\n" % int(len(loader_train)))
    # Build model
    model = NET(input_channel=32)
    # print_network(model)

    # loss function
    criterion = SSIM()
    criterion1 = nn.L1Loss()
    # Move to GPU
    if opt.use_GPU:
        model = model.cuda()
        criterion.cuda()
        criterion1.cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = MultiStepLR(optimizer, milestones=opt.milestone, gamma=0.2)
    # record training
    writer = SummaryWriter(opt.save_path)

    # load the lastest model
    initial_epoch = findLastCheckpoint(save_dir=opt.save_path)
    if initial_epoch > 0:
        print('resuming by loading epoch %d' % initial_epoch)
        model.load_state_dict(
            torch.load(os.path.join(opt.save_path, 'net_epoch%d.pth' % initial_epoch)))
        # start training
    step = 0
    for epoch in range(initial_epoch, opt.epochs):
        scheduler.step(epoch)
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])

        # epoch training start
        for i, (input_train, target_train) in enumerate(loader_train, 0):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            input_train, target_train = Variable(input_train), Variable(target_train)

            if opt.use_GPU:
                input_train, target_train = input_train.cuda(), target_train.cuda()

            out_train = model(input_train)
            pixel_metric = criterion(target_train, out_train)
            loss2 = criterion1(target_train, out_train)
            loss = 1 - pixel_metric + loss2

            loss.backward()
            optimizer.step()

            model.eval()
            out_train = model(input_train)
            out_train = torch.clamp(out_train, 0., 1.)
            psnr_train = batch_PSNR(out_train, target_train, 1.)
            print("[epoch %d][%d/%d] loss: %.4f， pixel_metric: %.4f，PSNR: %.4f" %
                  (epoch + 1, i + 1, len(loader_train), loss.item(), pixel_metric.item(), psnr_train))

            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1
        # epoch training end

        # log the images
        model.eval()
        out_train = model(input_train)
        out_train = torch.clamp(out_train, 0., 1.)
        im_target = utils.make_grid(target_train.data, nrow=8, normalize=True, scale_each=True)
        im_input = utils.make_grid(input_train.data, nrow=8, normalize=True, scale_each=True)
        im_derain = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
        writer.add_image('clean image', im_target, epoch + 1)
        writer.add_image('rainy image', im_input, epoch + 1)
        writer.add_image('deraining image', im_derain, epoch + 1)

        torch.save(model.state_dict(),
                   os.path.join(opt.save_path, 'net_latest.pth'))

        if epoch % opt.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_epoch%d.pth' % (epoch + 1)))


if __name__ == "__main__":
    prepare_data_Rain100L(data_path=opt.data_path, patch_size=128, stride=80)
    main()
