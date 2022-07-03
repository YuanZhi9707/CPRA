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
from DerainDataset import *
from utils import *
from torch.optim.lr_scheduler import MultiStepLR
from SSIM import *
from net import *
parser = argparse.ArgumentParser(description="Rain1200")
parser.add_argument("--preprocess", type=bool, default=False,help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=12, help="Training batch size")
parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=[30, 40, 50],
                    help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=0.0003, help="Initial learning rate")
parser.add_argument("--save_folder", type=str, default="logs/Rain1200", help='path of log files')
parser.add_argument("--save_freq", type=int, default=1, help='save intermediate model')
parser.add_argument("--data_path", type=str, default="datasets/train/Rain1200", help='path to training data')
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
opt = parser.parse_args()

if opt.use_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id


def main():
    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset(data_path=opt.data_path)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model

    model = NET(input_channel=32)

    criterion = SSIM()
    criterion1 = nn.L1Loss()
    # Move to GPU
    if opt.use_GPU:
        model = model.cuda()
        criterion.cuda()
        criterion1.cuda()
        # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = MultiStepLR(optimizer, milestones=opt.milestone, gamma=0.2)  # learning rates
    # training
    writer = SummaryWriter(opt.save_folder)
    step = 0

    initial_epoch = findLastCheckpoint(save_dir=opt.save_folder)  # load the last model in matconvnet style
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        model.load_state_dict(torch.load(os.path.join(opt.save_folder, 'net_epoch%d.pth' % initial_epoch)))

    for epoch in range(initial_epoch, opt.epochs):

        scheduler.step(epoch)
        # set learning rate
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])
        epoch_PSNR=0
        epoch_SSIM=0
        # train
        for i, (input, target) in enumerate(loader_train, 0):
            # training step

            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            if opt.use_GPU:
                input_train, target_train = Variable(input.cuda()), Variable(target.cuda())

            out_train = model(input_train)

            pixel_loss = criterion(target_train, out_train)
            loss2 = criterion1(target_train, out_train)
            loss = 1 - pixel_loss + loss2
            loss.backward()
            optimizer.step()
            # results
            model.eval()
            out_train = model(input_train)
            out_train = torch.clamp(out_train, 0., 1.)
            psnr_train = batch_PSNR(out_train, target_train, 1.)
            epoch_PSNR += psnr_train
            epoch_SSIM += pixel_loss.item()
            print("[epoch %d][%d/%d] loss: %.4f， pixel_metric: %.4f，PSNR: %.4f" %
                  (epoch + 1, i + 1, len(loader_train), loss.item(), pixel_loss.item(), psnr_train)
                  )
            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1

        print("epoch_PSNR: %.4f"%(epoch_PSNR/len(loader_train)))
        print("epoch_SSIM: %.4f"%(epoch_SSIM/len(loader_train)))
        # the end of each epoch

        model.eval()

        # log the images
        out_train = model(input_train)
        out_train = torch.clamp(out_train, 0., 1.)  # 控制输出在（0-1）之间
        Img = utils.make_grid(target_train.data, nrow=8, normalize=True, scale_each=True)  # 合并图像
        Imgn = utils.make_grid(input_train.data, nrow=8, normalize=True, scale_each=True)
        Irecon = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
        writer.add_image('clean image', Img, epoch)
        writer.add_image('noisy image', Imgn, epoch)
        writer.add_image('reconstructed image', Irecon, epoch)
        # save model
        torch.save(model.state_dict(), os.path.join(opt.save_folder, 'net_latest.pth'))

        if epoch % opt.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(opt.save_folder, 'net_epoch%d.pth' % (epoch + 1)))
        print("[epoch %d] , avg_ssim: %.4f,avg_psnr: %.4f" %
                (epoch + 1, epoch_SSIM / len(loader_train), epoch_PSNR / len(loader_train)))
        f = open('./train_save.txt',mode='a')
        f.write('epoch:%d' % (epoch + 1) + '     ')
        f.write('avg_ssim:%2.4f'%(epoch_SSIM / len(loader_train))+'     ')
        f.write('avg_psnr:%2.4f'%(epoch_PSNR / len(loader_train))+'     '+'\n')
        f.close()


if __name__ == "__main__":
    if opt.preprocess:
        prepare_data_Rain1400(data_path=opt.data_path, patch_size=128, stride=80)

    main()
