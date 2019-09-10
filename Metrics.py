import numpy as np
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr

def PSNRV2(img1, img2):
    target = np.array(img1)
    output = np.array(img2)
    my_psnrv2 = psnr(target, output)
    return my_psnrv2


def SSIM(img1, img2):
    target = np.array(img1)
    output = np.array(img2)
    my_ssim = ssim(target, output, multichannel=True)
    return my_ssim
