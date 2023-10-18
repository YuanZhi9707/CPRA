# -*- coding: utf-8 -*-
# =============================================================================
# run this to result the model
import argparse
from torch.autograd import Variable
from DerainDataset import *
from DerainDataset import normalize
from net import NET
from skimage.metrics import peak_signal_noise_ratio,structural_similarity
# from DAB_sep_mwt import DenoiseNet


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='./datasets/test/Rain1200/rainy', required=False,
                    type=str, help='Directory of train images')
parser.add_argument('--target_dir', default='./datasets/test/Rain1200/norain', required=False,
                    type=str, help='Directory of test images')
parser.add_argument('--save_dir', default='./results/Rain1200',
                    type=str, help='Directory for results')
parser.add_argument('--weights', default='./logs/Rain1200/net_latest.pth', required=False,
                    type=str, help='Path to weights')
parser.add_argument('--test_select', default='test_rain1200', required=False,
                    type=str, help='Function selection')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--use_GPU', default=True, type=str, help='CUDA_VISIBLE_DEVICES')
args = parser.parse_args()

def test_rain1200():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    model = NET()
    if args.use_GPU:
        model1 = model.cuda()
    model.load_state_dict(torch.load(args.weights))
    model.eval()

    count = 0
    psnr_val_rgb = []
    SSIM_VAL = []

    input_files = os.listdir(args.input_dir)
    target_files = os.listdir(args.target_dir)
    for i in range(100):
        img_path = os.path.join(args.input_dir, input_files[i])
        target_path = os.path.join(args.target_dir, target_files[i])
        target = cv2.imread(target_path)
        H,W,C = target.shape
        index = target_files[i].split('.')
        y = cv2.imread(img_path)

        b, g, r = cv2.split(y)
        y = cv2.merge([r, g, b])
        y = normalize(np.float32(y))
        y = np.expand_dims(y.transpose(2, 0, 1), 0)
        y = Variable(torch.Tensor(y))
        if args.use_GPU:
            y = y.cuda()
        with torch.no_grad():
            if args.use_GPU:
                torch.cuda.synchronize()
                out = model(y)

        out = torch.clamp(out, 0., 1.)
        if args.use_GPU:
            save_out = np.uint8(255 * out.data.cpu().numpy().squeeze())  # back to cpu
        else:
            save_out = np.uint8(255 * out.data.numpy().squeeze())
        save_out = save_out.transpose(1, 2, 0)
        b, g, r = cv2.split(save_out)
        save_out = cv2.merge([r, g, b])

        cv2.imwrite(os.path.join(args.save_dir,'derain-%s'%str(index[0])+'.png'), save_out)
        count += 1
        psnr_val_rgb.append(peak_signal_noise_ratio(save_out, target, data_range=255.))
        SSIM_VAL.append(structural_similarity(save_out,target,multichannel=True))
        print("image%s:" % str(i) + '    ' + 'PSNR:%2.4f' % psnr_val_rgb[i] + '    ' + 'SSIM:%2.4f' % SSIM_VAL[i]+'    ')
    psnr_val_rgb = sum(psnr_val_rgb) / len(psnr_val_rgb)
    SSIM_VAL=sum(SSIM_VAL)/len(SSIM_VAL)
    print("PSNR: %.4f " % (psnr_val_rgb))
    print("SSIM: %.4f"%(SSIM_VAL))

def test_rain12():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    model = NET()
    if args.use_GPU:
        model1 = model.cuda()
    model.load_state_dict(torch.load(args.weights))
    model.eval()

    count = 0
    psnr_val_rgb = []
    SSIM_VAL = []

    input_files = os.listdir(args.input_dir)
    target_files = os.listdir(args.target_dir)
    for i in range(100):
        img_path = os.path.join(args.input_dir, input_files[i])
        target_path = os.path.join(args.target_dir, target_files[i])
        target = cv2.imread(target_path)
        #1400_test
        H,W,C = target.shape
        index = target_files[i].split('.')

        y = cv2.imread(img_path)

        b, g, r = cv2.split(y)
        y = cv2.merge([r, g, b])
        y = normalize(np.float32(y))
        y = np.expand_dims(y.transpose(2, 0, 1), 0)
        y = Variable(torch.Tensor(y))
        if args.use_GPU:
            y = y.cuda()
        with torch.no_grad():
            if args.use_GPU:
                torch.cuda.synchronize()
                out = model(y)

        out = torch.clamp(out, 0., 1.)
        if args.use_GPU:
            save_out = np.uint8(255 * out.data.cpu().numpy().squeeze())  # back to cpu
        else:
            save_out = np.uint8(255 * out.data.numpy().squeeze())
        save_out = save_out.transpose(1, 2, 0)
        b, g, r = cv2.split(save_out)
        save_out = cv2.merge([r, g, b])

        cv2.imwrite(os.path.join(args.save_dir,'derain-%s'%str(index[0])+'.png'), save_out)
        count += 1
        psnr_val_rgb.append(peak_signal_noise_ratio(save_out, target, data_range=255.))
        SSIM_VAL.append(structural_similarity(save_out,target,multichannel=True))
        print("image%s:" % str(i) + '    ' + 'PSNR:%2.4f' % psnr_val_rgb[i] + '    ' + 'SSIM:%2.4f' % SSIM_VAL[i]+'    ')
    psnr_val_rgb = sum(psnr_val_rgb) / len(psnr_val_rgb)
    SSIM_VAL=sum(SSIM_VAL)/len(SSIM_VAL)
    print("PSNR: %.4f " % (psnr_val_rgb))
    print("SSIM: %.4f"%(SSIM_VAL))

def test_real(data_path,save_path):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    input_files = os.listdir(data_path)
    for i in range(55):
        img_path = os.path.join(data_path, input_files[i])
        input = cv2.imread(img_path)

        model = NET()
        model = model.cuda()
        model.load_state_dict(torch.load(args.weights))
        model.eval()

        input = normalize(np.float32(input))
        input = np.expand_dims(input.transpose(2, 0, 1), 0)
        input = Variable(torch.Tensor(input)).cuda()
        with torch.no_grad():
            out = model(input)
            save_out = 255 * out.data.cpu().numpy().squeeze()  # back to cpu
            save_out = save_out.transpose(1, 2, 0)
            cv2.imwrite(os.path.join(save_path, input_files[i]), save_out)

if __name__ == '__main__':
    print("start running"+args.test_select+"()")
    test_rain1200()

