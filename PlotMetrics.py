import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataLRS=pd.read_csv('/media/student/SPATIU/LTSID/Metrics/metrics_C.csv', sep=' ', header=None)
data=pd.read_csv('/media/student/SPATIU/LTSID/Metrics/metrics_D.csv', sep=" ", header=None)
psnr_C=dataLRS[1].values
ssim_C=dataLRS[3].values
psnr_D=data[1].values
ssim_D=data[3].values

avg_C=np.mean(psnr_C)
avg_D=np.mean(psnr_D)
plt.xlabel('Photo')
plt.ylabel('PSNR')
plt.title('PSNR_Classic VS. PSNR_Dropout')
line_up, = plt.plot(np.array(range(0, len(psnr_C))), psnr_C, label='PSNR Classic avg: %0.5f ' % (avg_C))
line_down, = plt.plot(np.array(range(0, len(psnr_D))), psnr_D, label='PSNR Dropout avg: %0.5f ' % (avg_D))
plt.legend(handles=[line_up, line_down])

plt.show()

minLRS=np.mean(ssim_C)
minOri=np.mean(ssim_D)
plt.xlabel('Photo')
plt.ylabel('PSNR')
plt.title('SSIM_Classic VS. SSIM_Dropout')
line_up, = plt.plot(np.array(range(0, len(ssim_C))), ssim_C, label='SSIM Classic avg: %0.5f ' % (minLRS))
line_down, = plt.plot(np.array(range(0, len(ssim_D))), ssim_D, label='SSIM Dropout avg: %0.5f ' % (minOri))
plt.legend(handles=[line_up, line_down])

plt.show()
