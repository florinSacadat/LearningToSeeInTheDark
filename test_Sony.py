import os
import numpy as np
import VisdomUtils as V
import torch
import Metrics
import ToImage
from torch.utils.data import DataLoader
import Model as M
from DatasetTest import SID

input_dir = './dataset/Sony/short/'
gt_dir = './dataset/Sony/long/'
m_path = '/media/student/SPATIU/LTSID/saved_model/'
m_name = 'checkpoint_sony_e0045D'
result_dir = './test_result/D-45/'

device = torch.device('cpu')

path_input_txt = '/media/student/SPATIU/LTSID/Tensor_Test_Image/best_test_input_tensors.txt'
path_target_txt = '/media/student/SPATIU/LTSID/Tensor_Test_Image/best_target_test_tensors.txt'
path_input_images = '/media/student/SPATIU/LTSID/Tensor_Test_Image/Input/'
path_target_images = '/media/student/SPATIU/LTSID/Tensor_Test_Image/Target/'
scale_path='/media/student/SPATIU/LTSID/Tensor_Test_Image/Scale/'
test_set = SID(path_input_txt, path_target_txt, path_input_images, path_target_images,scale_path)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)



model = M.SeeInDarkDrop().cpu()
model.load_state_dict(torch.load(m_path + m_name))
if not os.path.isdir(result_dir):
    os.makedirs(result_dir)

plotter =V.VisdomLinePlotter('test_Sony')
plot2=V.VisdomLinePlotter('test_Sony')
model.eval()
for batch_idx, (inputs, targets, scales) in enumerate(test_loader):
    inputs, targets, scales = inputs.cpu(), targets.cpu(), scales.cpu()

    print(batch_idx)
    inputs = np.minimum(inputs, 1.0)

    in_img = inputs.permute(0, 3, 1, 2).cpu()
    with torch.no_grad():
        out_img = model(in_img)
    output = out_img.permute(0, 2, 3, 1).data.numpy()
    output = np.minimum(np.maximum(output, 0), 1)

    output = output[0, :, :, :]
    gt_full = targets[0, :, :, :]
    scale_full = scales[0, :, :, :]
    origin_full = scale_full
    scale_full = scale_full * torch.mean(gt_full) / torch.mean(
        scale_full)  # scale the low-light image to the same mean of the groundtruth

    # calculate the metrics
    psnr_var = Metrics.PSNRV2(gt_full, output)
    print("PSNR: "+str(psnr_var))
    plotter.plot("loss", "PSNR", "PSNR", batch_idx, psnr_var)
    ssim_var=Metrics.SSIM(gt_full,output)
    print("SSIM: "+ str(ssim_var))
    plot2.plot("loss", "SSIM", "SSIM", batch_idx, ssim_var)

    # save teh log
    with open("/media/student/SPATIU/LTSID/Metrics/metrics_D.csv", "a") as log:
            log.write("{0} {1}\n".format('PSNR: ' + str(psnr_var), 'SSIM: ' + str(ssim_var)))
    # save in folder the imgs
    if batch_idx%10==0:
        ToImage.toimage(origin_full * 255, high=255, low=0, cmin=0, cmax=255).save(
            result_dir + '%d_00_ori.png' % batch_idx)
        ToImage.toimage(output * 255, high=255, low=0, cmin=0, cmax=255).save(
            result_dir + '%d_00_out.png' % batch_idx)
        ToImage.toimage(scale_full * 255, high=255, low=0, cmin=0, cmax=255).save(
            result_dir + '%d_00_scale.png' % batch_idx)
        ToImage.toimage(gt_full * 255, high=255, low=0, cmin=0, cmax=255).save(
            result_dir + '%d_00_gt.png' % batch_idx)

model.train()