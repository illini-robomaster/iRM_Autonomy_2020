 
import os
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
#%matplotlib widget
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from progressbar import ProgressBar

fg_path = '/home/xiaoboh2/rm_synethesis_hal/data/Synethesized_Dataset/dataset_out/image_out/*'
#label_path='./models/Batch Renderding/scene/label_out/*'
bg_paths = ['/home/shared/imagenet/raw/val_nodir/']+glob.glob('/home/shared/imagenet/raw/train/*')


fg_seg_pairs=[]
def crop_zero(image,reference=None):
    if(reference is None):
        reference=image
    nonzeros = cv2.findNonZero(reference[:,:,-1])
    upper = np.squeeze(np.max(nonzeros, axis=0).astype(int))
    lower = np.squeeze(np.min(nonzeros, axis=0).astype(int))
    return image[lower[1]:upper[1],lower[0]:upper[0]]

pbar = ProgressBar()
for name in pbar((glob.glob(fg_path))):
    image = cv2.imread(name,cv2.IMREAD_UNCHANGED)
    if(image is None): 
        #print('Failed to load '+name)
        continue
    image[:, :, 0], image[:,:, 2] = image[:,:, 2], image[:, :,0].copy()
    label=cv2.imread(name.replace('image','label'),cv2.IMREAD_UNCHANGED)
    if(label is None): 
        #print('Failed to load '+name)
        continue
    label=crop_zero(label,reference=image)
    if(np.sum(label)==0):continue
    #label=np.sum(label,axis=2)
    #label[label>0]=1
    image=crop_zero(image)
    fg_seg_pairs+=[[image,label]]
    '''
    plt.figure()
    plt.imshow(image)
    plt.imshow(label)
    break
    '''
print('{} pairs of foreground loaded.'.format(fg_seg_pairs.__len__()))

res = [ 1080,960]
# size ratio range, numbers, blur, shear, explosure
para_space_range = {'size': [0.15, 0.6], 'min_num': [1, 2],'min_area':[0.001,0.002]}


def get_para():
    para = para_space_range.copy()
    para['min_num'] = np.random.uniform(low=para_space_range['min_num'][0], high=para_space_range['min_num'][1])
    para['size'] = np.random.uniform(low=para_space_range['size'][0], high=para_space_range['size'][1],
                                     size=int(para['min_num'])*20)
    para['min_area']=np.random.uniform(low=para_space_range['min_area'][0], high=para_space_range['min_area'][1])
    return para


def get_pair_PIL():
    idx = np.random.randint(0, fg_seg_pairs.__len__())
    fg_pair = fg_seg_pairs[idx]
    fg=fg_pair[0].copy()
    #fg[:,:,0:2]=0
    #fg[:,:,3][fg[:,:,2]<140]=0
    fg = Image.fromarray(fg, 'RGBA')
    label = Image.fromarray(fg_pair[1], 'RGBA')
    return fg, label

bg_files=[]
for bg_path in bg_paths: bg_files+=glob.glob(bg_path+'*')
print('{} backgrounds found. '.format(bg_files.__len__()))

buffer=[]
buffered_files=[]
def get_bg_pair():
    #print(buffer.__len__())
    files=bg_files
    idx = np.random.randint(0, bg_path.__len__())
    if(files[idx] in buffered_files):
        idx=np.random.randint(0,buffer.__len__())
        bg=buffer[idx][0]
        bg_label=buffer[idx][1]
    else:
        bg = Image.open(files[idx])
        bg = bg.resize(res)
        bg_label=Image.new('RGBA', bg.size, (0, 0, 0, 0))
        buffer.append((bg,bg_label))
        buffered_files.append(files[idx])
    return bg.copy(),bg_label.copy()

def area_percent(img):
    img_full=img.copy()
    img_full[:]=255
    return np.sum(img)/np.sum(img_full)

#pbar = ProgressBar()
#for bg in pbar(glob.glob(bg_path)):
def get_blended(plot=True):
    bg,bg_label=get_bg_pair()
    para = get_para()
    n=0
    while area_percent(np.array(bg_label))<para['min_area'] or n<=np.min(para['min_num']):
    #for n in range(int(para['min_num'])):
        fg, label = get_pair_PIL()
        newsize=(np.array(res)*para['size'][n%int(para['size'].shape[0])]).astype(np.int)
        fg = fg.resize(newsize)
        label = label.resize(newsize)
        
        loc = (np.random.randint(0, res[0]), np.random.randint(0, res[1]))
        bg.paste(fg, loc, fg)
        bg_label.paste(label,loc,label)
        n+=1

    if plot:
        print(para)
        plt.figure(figsize=(15, 15))
        plt.imshow(bg)
        plt.imshow(bg_label)
    return bg,bg_label
get_blended()


import torch
import torch.nn as nn
import torch.nn.functional as F


def downconv(in_channels, out_channels, kernel_size):
    padding = int(kernel_size / 2)
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                  kernel_size=kernel_size, padding=padding),
        nn.ReLU(),
    )


def downsamp(channels):
    return nn.Sequential(
        nn.MaxPool2d(2, 2),
        nn.BatchNorm2d(channels)
    )


def up(in_channels, out_channels, kernel_size):
    padding = int(kernel_size / 2)
    return nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                  kernel_size=kernel_size, padding=padding),
        nn.ReLU(),
        nn.Upsample(scale_factor=2)
    )


# %%
class Model(nn.Module):
    def __init__(self, in_channels=3, channels=64, kernel_size=3, dropout=0.5):
        super(Model, self).__init__()
        self.dropout = dropout
        self.downconv1 = downconv(in_channels, channels, kernel_size)
        self.downsamp1 = downsamp(channels)
        self.downconv2 = downconv(channels, 2 * channels, kernel_size)
        self.downsamp2 = downsamp(2 * channels)
        self.downconv3 = downconv(2 * channels, 4 * channels, kernel_size)
        self.downsamp3 = downsamp(4 * channels)
        self.up1 = up(4 * channels, 4 * channels, kernel_size)
        self.up2 = up(8 * channels, 2 * channels, kernel_size)
        self.up3 = up(4 * channels, channels, kernel_size)
        padding = int(kernel_size / 2)
        self.last = nn.Sequential(
            nn.Conv2d(in_channels=2 * channels, out_channels=channels,
                      kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels,
                      kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=1)#,
            #nn.Sigmoid()
        )

    def forward(self, x):
        d1 = self.downconv1(x)
        ds1 = self.downsamp1(d1)
        d2 = self.downconv2(ds1)
        ds2 = self.downsamp2(d2)
        d3 = self.downconv3(ds2)
        ds3 = self.downsamp3(d3)
        u = self.up1(ds3)
        u = torch.cat((d3, u), dim=1)
        u = nn.Dropout2d(self.dropout)(u)
        u = self.up2(u)
        u = torch.cat((d2, u), dim=1)
        u = nn.Dropout2d(self.dropout)(u)
        u = self.up3(u)
        u = torch.cat((d1, u), dim=1)
        u = nn.Dropout2d(self.dropout)(u)
        u = self.last(u)
        u = u.reshape(-1, res[1], res[0])
        return u
model=Model()
#np.random.seed()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # use GPU if available
model.to(device)
model = nn.DataParallel(model)
print(device)

batch_size=8
def load_train(queue):
    while True:
        img_batch=[]
        label_batch=[]
        try:
            while img_batch.__len__()<batch_size:
                img,label=get_blended(plot=False)
                img=np.array(img)[:,:,0:4]
                label=np.array(label)
                label=np.sum(label,axis=2)
                label[label!=0]=1
                img_batch.append(img.astype(np.float))
                label_batch.append(label.astype(np.float))
        except Exception as e:
                print(e)
                continue
        img_batch=np.array(img_batch).transpose([0,3,1,2])
        label_batch=np.array(label_batch).transpose([0,1,2])
        queue.put((img_batch,label_batch))
        #print('load')
        
q_train = torch.multiprocessing.Queue(maxsize=100)
for i in range(1):
    p1 = torch.multiprocessing.Process(target=load_train, args=(q_train,))
#load_train(q_train)
p1.start()

save_directory_name = './checkpoint'
sample_directory_name = './samples'

optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
criterion = nn.BCEWithLogitsLoss().to(device)
for e in range(10000000):
    model.train()
    pair = q_train.get()
    img = torch.tensor(pair[0],dtype=torch.float, requires_grad = True,device=device)
    label = torch.tensor(pair[1],dtype=torch.float,device=device)
    optimizer.zero_grad()
    pred = model(img)
    loss = criterion(pred.reshape([batch_size,-1]), label.reshape([batch_size,-1]))
    loss.backward()
    optimizer.step()
    print(loss)
    
    if e%64 == 0:
        np.savez_compressed(sample_directory_name+'/sample', pred.cpu().data.numpy())
        np.savez_compressed(sample_directory_name+'/sample_img', img.cpu().data.numpy())
        np.savez_compressed(sample_directory_name+'/sample_label', label.cpu().data.numpy())
        torch.save(model.state_dict(), save_directory_name+'/model.pth')
        print('Checkpoint saved.')
