from img_utils import *
# Select here between unet/fcn for which arch is used
#from unet import *
from fcn import *
from progressbar import ProgressBar
from PIL import Image
import cv2
import torch
import os
import time
import glob
import numpy as np
np.random.seed(0)

batch_size = 32

model = Model()
device = torch.device("cuda:0" if torch.cuda.is_available()
                      else "cpu")  # use GPU if available
model.to(device)
model = nn.DataParallel(model)
print(device)
print('Model Init ok')


def load_train(queue):
    while True:
        img_batch = []
        label_batch = []
        while img_batch.__len__() < batch_size:
            try:
                img, label = get_blended(plot=False, augment=True)
                img = np.array(img)
                if(len(img.shape) != 3 or img.shape[2] != 3):
                    continue
                label = np.array(label)
                label = np.sum(label, axis=2, keepdims=True)
                label[label != 0] = 1
                img_batch.append(img.astype(np.float))
                label_batch.append(label.astype(np.float))
            except Exception as e:
                print(e)
                continue
        img_batch = np.array(img_batch)
        img_batch = img_batch.transpose([0, 3, 1, 2])
        img_batch[img_batch != 0] /= 255
        label_batch = np.array(label_batch)
        label_batch = label_batch.transpose([0, 3, 1, 2])
        queue.put((img_batch, label_batch))


q_train = torch.multiprocessing.Queue(maxsize=200)
for i in range(16):
    p1 = torch.multiprocessing.Process(target=load_train, args=(q_train,))
    p1.start()

save_directory_name = './checkpoint'
sample_directory_name = './samples'

optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-4)
criterion = nn.MSELoss().to(device)
for e in range(10000000):
    model.train()
    pair = q_train.get()
    img = torch.tensor(pair[0], dtype=torch.float,
                       requires_grad=True, device=device)
    label = torch.tensor(pair[1], dtype=torch.float, device=device)

    optimizer.zero_grad()
    pred = model(img)
    loss = criterion(pred.reshape([batch_size, -1]),
                     label.reshape([batch_size, -1]))
    loss.backward()
    optimizer.step()

    if e % 8 == 0:
        print(f'Iteration: {e}, Loss: {loss}')
    if e % 512 == 511:
        np.savez_compressed(sample_directory_name +
                            '/sample', pred.cpu().data.numpy())
        np.savez_compressed(sample_directory_name +
                            '/sample_img', img.cpu().data.numpy())
        np.savez_compressed(sample_directory_name +
                            '/sample_label', label.cpu().data.numpy())
        torch.save(model.state_dict(), save_directory_name+'/model.pth')
        print('Checkpoint saved.')
