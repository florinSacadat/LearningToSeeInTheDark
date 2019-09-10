
import os
import numpy as np
import glob
import torch
import torch.optim as optim
import ToImage
from torch.utils.data import DataLoader
import Model as M
import Dataset as D
import VisdomUtils

input_dir = '/media/student/SPATIU/LTSID/dataset/Sony/short/'
gt_dir = '/media/student/SPATIU/LTSID/dataset/Sony/long/'
model_dir = '/media/student/SPATIU/LTSID/saved_model/'
result_dir = './result_Sony/'
m_name = 'checkpoint_sony_e4000.pth'
my_saved_model = '/media/student/SPATIU/LTSID/Last model/MODEL_C'
device = torch.device('cuda:0')

# Train paths and Load

path_input_txt = '/media/student/SPATIU/LTSID/TensorImage/best_input_val_tensors.txt'
path_target_txt = '//media/student/SPATIU/LTSID/TensorImage/best_target_val_tensors.txt'
path_input_images = '/media/student/SPATIU/LTSID/TensorImage/Input/'
path_target_images= '/media/student/SPATIU/LTSID/TensorImage/Target/'

train_set = D.SeeInDark(path_input_txt, path_target_txt, path_input_images, path_target_images)
train_loader = DataLoader(train_set, batch_size=1, shuffle=True)

# Val paths and Load
path_val_input_txt = '/media/student/SPATIU/LTSID/TensorImage/best_input_val_tensors.txt'
path_val_target_txt = '//media/student/SPATIU/LTSID/TensorImage/best_target_val_tensors.txt'
path_val_input_images = '/media/student/SPATIU/LTSID/TensorImage/Input/'
path_val_target_images= '/media/student/SPATIU/LTSID/TensorImage/Target/'
val_set = D.SeeInDark(path_val_input_txt, path_val_target_txt, path_val_input_images, path_val_target_images)
val_loader = DataLoader(val_set, batch_size=1, shuffle=True)

def valid_loss_function(model, val_loader):
    g_loss = np.zeros((5000, 1))
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        inputs, targets = inputs, targets
        inputs = np.minimum(inputs, 1.0)
        targets = np.maximum(targets, 0.0)

        in_img = inputs.permute(0, 3, 1, 2).cuda()
        gt_img = targets.permute(0, 3, 1, 2).cuda()

        out_img = model(in_img)
        out_img = out_img.cuda()

        loss = reduce_mean(out_img, gt_img)

        g_loss[batch_idx] = loss.data.cpu()

    return np.mean(g_loss[np.where(g_loss)])

def reduce_mean(out_im, gt_im):
    return torch.abs(out_im - gt_im).mean()

ps = 512  # patch size for training
save_freq = 5

g_loss = np.zeros((5000, 1))

allfolders = glob.glob('./result_Sony/*0')
lastepoch = 0
for folder in allfolders:
    lastepoch = np.maximum(lastepoch, int(folder[-4:]))

learning_rate = 1e-4
model = M.SeeInDark().cuda()
model._initialize_weights()
# model.load_state_dict(torch.load(my_saved_model))
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#
plotter1 = VisdomUtils.VisdomLinePlotter(env_name='Train')
plotter2 = VisdomUtils.VisdomLinePlotter(env_name='Train')

final_loss = 0
for epoch in range(lastepoch, 46):
    model.train()
    print('Epoch:' + str(epoch))
    if os.path.isdir("result/%04d" % epoch):
        continue

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs, targets
        inputs = np.minimum(inputs, 1.0)
        targets = np.maximum(targets, 0.0)

        in_img = inputs.permute(0, 3, 1, 2).cuda()
        gt_img = targets.permute(0, 3, 1, 2).cuda()

        optimizer.zero_grad()
        out_img = model(in_img)
        out_img = out_img.cuda()

        loss = reduce_mean(out_img, gt_img)
        loss.backward()

        optimizer.step()
        g_loss[batch_idx] = loss.data.cpu()

        final_loss = np.mean(g_loss[np.where(g_loss)])
        print("%d %d Loss=%.10f" % (epoch, batch_idx, final_loss))
        # plotter2.plot('loss', 'Batch - Loss', 'Batch Idx', batch_idx, final_loss)


        if epoch % save_freq == 0 :
            if not os.path.isdir(result_dir + '%04d' % epoch+'C'):
                os.makedirs(result_dir + '%04d' % epoch+'C')
            output = out_img.permute(0, 2, 3, 1).cpu().data.numpy()
            output = np.minimum(np.maximum(output, 0), 1)

            temp = np.concatenate((targets[0, :, :, :], output[0, :, :, :]), axis=1)
            ToImage.toimage(temp * 255, high=255, low=0, cmin=0, cmax=255).save(
                result_dir + '%04dC/_00_train_%d.jpg' % (epoch, batch_idx))
            torch.save(model.state_dict(), model_dir + 'checkpoint_sony_e%04d' % epoch+'C')

        if epoch % save_freq == 0 and batch_idx == (len(train_loader) - 1):
            my_val_loss = valid_loss_function(model, val_loader)
            print("Validation Loss=%.10f" % my_val_loss)
            plotter1.plot('loss', 'Validation Loss', 'Epoch', epoch, my_val_loss)

    plotter1.plot('loss', 'Loss', 'Epoch', epoch, final_loss)

    torch.save(model.state_dict(),
               '/media/student/SPATIU/LTSID/Last model/MODEL_C')
