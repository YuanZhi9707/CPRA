# -*- coding: utf-8 -*-
# =============================================================================
# run this to result the model
import argparse
from utils import *
from torch.autograd import Variable
from DerainDataset import *
from DerainDataset import normalize
from MDARNet import MDARNet
# from DAB_sep_mwt import DenoiseNet
from Ablation.MDARNet_noMWT import DenoiseNet
# from Ablation.MDARNet_RG5 import DenoiseNet
import lpips
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='./data/rain100H/rainy', required=False,
                    type=str, help='Directory of train images')
parser.add_argument('--target_dir', default='./data/rain100H/norain', required=False,
                    type=str, help='Directory of test images')
parser.add_argument('--save_dir', default='./result',
                    type=str, help='Directory for results')
parser.add_argument('--weights', default='./models/rain100H.pth', required=False,
                    type=str, help='Path to weights')
parser.add_argument('--test_select', default='test_rain100H', required=False,
                    type=str, help='Function selection')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--use_GPU', default=True, type=str, help='CUDA_VISIBLE_DEVICES')
args = parser.parse_args()

def test_rain1800H():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    model1 = MDARNet(H=321, W=481, batch_size=1)
    model2 = MDARNet(H=481, W=321, batch_size=1)
    if args.use_GPU:
        model1 = model1.cuda()
        model2 = model2.cuda()
    model1.load_state_dict(torch.load(args.weights))
    model2.load_state_dict(torch.load(args.weights))
    model1.eval()
    model2.eval()

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
            if H == 321 and W == 481:
                out = model1(y)
            if H == 481 and W == 321:
                out = model2(y)

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
def test_rain1800L():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    model1 = MDARNet(H=321, W=481, batch_size=1)
    model2 = MDARNet(H=481, W=321, batch_size=1)
    if args.use_GPU:
        model1 = model1.cuda()
        model2 = model2.cuda()
    model1.load_state_dict(torch.load(args.weights))
    model2.load_state_dict(torch.load(args.weights))
    model1.eval()
    model2.eval()
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
            if H == 321 and W == 481:
                out = model1(y)
            if H == 481 and W == 321:
                out = model2(y)

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
    model1 = MDARNet(H=321, W=481, batch_size=1)
    model2 = MDARNet(H=481, W=321, batch_size=1)
    if args.use_GPU:
        model1 = model1.cuda()
        model2 = model2.cuda()
    model1.load_state_dict(torch.load(args.weights))
    model2.load_state_dict(torch.load(args.weights))
    model1.eval()
    model2.eval()
    count = 0
    psnr_val_rgb = []
    SSIM_VAL = []
    input_files = os.listdir(args.input_dir)
    target_files = os.listdir(args.target_dir)
    for i in range(12):
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
            if H == 321 and W == 481:
                out = model1(y)
            if H == 481 and W == 321:
                out = model2(y)
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
def test_rain1200():
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    model = MDARNet(H=512, W=512, batch_size=1)
    if args.use_GPU:
        model = model.cuda()
    model.load_state_dict(torch.load(args.weights))
    model.eval()

    count = 0
    psnr_val_rgb = []
    SSIM_VAL = []

    input_files = os.listdir(args.input_dir)
    target_files = os.listdir(args.target_dir)
    for i in range(120):
        img_path = os.path.join(args.input_dir, input_files[i])
        target_path = os.path.join(args.target_dir, target_files[i])
        target = cv2.imread(target_path)
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
            if args.use_GPU:
                torch.cuda.synchronize()
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
def test_real(data_path,save_path,weight_path):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    input_files = os.listdir(data_path)
    for i in range(55):
        img_path = os.path.join(data_path, input_files[i])
        input = cv2.imread(img_path)

        model = MDARNet(H=481, W=321, batch_size=1)
        model = model.cuda()
        model.load_state_dict(torch.load(weight_path))
        model.eval()

        input = normalize(np.float32(input))
        input = np.expand_dims(input.transpose(2, 0, 1), 0)
        input = Variable(torch.Tensor(input)).cuda()
        with torch.no_grad():
            out = model(input)
            save_out = 255 * out.data.cpu().numpy().squeeze()  # back to cpu
            save_out = save_out.transpose(1, 2, 0)
            cv2.imwrite(os.path.join(save_path, input_files[i]), save_out)
def compute():
    data_len = 100
    norain_path = './result/Ablation/4/M1/norain'
    rain_path = 'result/Ablation/4/M1/derain'
    target_file = os.listdir(norain_path)
    output_file = os.listdir(rain_path)
    psnr = []
    ssim = []
    lpips_val = []
    lpips_loss = lpips.LPIPS(net='alex')
    lpips_loss.eval()
    for i in range(data_len):
        target_path = os.path.join(norain_path,target_file[i])
        output_path = os.path.join(rain_path,output_file[i])
        target = cv2.imread(target_path)
        output = cv2.imread(output_path)
        h,w,c = output.shape
        target = target[0:h,0:w,:]
        target_copy = target.copy()
        output_copy = output.copy()
        target_copy = np.expand_dims(target_copy.transpose(2, 0, 1), 0)
        output_copy = np.expand_dims(output_copy.transpose(2, 0, 1), 0)
        target_copy = torch.Tensor(target_copy)
        output_copy = torch.Tensor(output_copy)

        psnr.append(peak_signal_noise_ratio(output, target, data_range=255.))
        ssim.append(structural_similarity(output,target,multichannel=True))
        lpips_val.append(lpips_loss(output_copy, target_copy))
        print("image%s:" % str(i+1) + '    ' + 'PSNR:%2.4f' % psnr[i] + '    ' + 'SSIM:%2.4f' % ssim[i]+'    '+'LPIPS:%2.4f'%lpips_val[i])
    avg_psnr = sum(psnr)/data_len
    avg_ssim = sum(ssim)/data_len
    avg_lpips = sum(lpips_val)/data_len
    print("PSNR: %.4f " % (avg_psnr))
    print("SSIM: %.4f" % (avg_ssim))
    print('LPIPS: %.4f' % (avg_lpips))

if __name__ == '__main__':
    print("start running"+args.test_select+"()")
    if args.test_select == "test_rain100H":
        test_rain1800H()
    if args.test_select == "test_rain100L":
        test_rain1800L()
    if args.test_select == "test_rain1200":
        test_rain1200()
    if args.test_select == "test_rain12":
        test_rain12()

