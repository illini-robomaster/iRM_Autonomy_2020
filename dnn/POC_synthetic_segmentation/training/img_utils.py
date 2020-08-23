import imgaug as ia
import imgaug.augmenters as iaa
import torch
import os
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import matplotlib
from PIL import Image
from progressbar import ProgressBar

is_HAL = len(glob.glob('/home/shared/imagenet/raw/val_nodir/'))
if not is_HAL:
    fg_path = './models/Batch Renderding/renders/scene_horizontal_no_number/image_out/*'
    bg_paths = ['./DATA/data/imagenet/raw/val_nodir/**']
    fg_num = 64
else:  # On server
    fg_path = './data/Synethesized_Dataset/data_set_horizontal_new/image_out/*'
    fg_num = -1

    bg_paths = ['./data/ade20k/ade20k_nodir/*']


def crop_zero(image, reference=None):
    if(reference is None):
        reference = image
    nonzeros = cv2.findNonZero(reference[:, :, -1])
    upper = np.squeeze(np.max(nonzeros, axis=0).astype(int))
    lower = np.squeeze(np.min(nonzeros, axis=0).astype(int))
    return image[lower[1]:upper[1], lower[0]:upper[0]]


def remove_too_small(label, min_size=5000):
    # Remove armor places with too small visible area
    label = label.copy()
    mask = np.sum(label, axis=2).astype(np.uint8)
    mask[mask != 0] = 255
    # This will get stuck, disable for now
    # nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
    # mask.copy(), connectivity=8)
    # cv2.connectedComponentsWithStats(mask.copy(), connectivity=4)
    # sizes = stats[1:, -1]
    # mask = np.zeros_like(mask)
    # for i in range(0, nb_components-1):
    # if sizes[i] >= min_size:
    # mask[output == i + 1] = 255
    kernel = np.ones((50, 50), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)
    label[:, :, 0][mask == 0] = 0
    label[:, :, 1][mask == 0] = 0
    label[:, :, 2][mask == 0] = 0
    label[:, :, 3][mask == 0] = 0
    return label


def load_foreground():
    fg_seg_pairs = []
    print(f'Loading forgrounds from {fg_path}:')
    files = glob.glob(fg_path)
    files.sort()
    for name in ProgressBar()(files[0:min(len(files), fg_num)]):
        try:
            image = cv2.imread(name, cv2.IMREAD_UNCHANGED)
            image[:, :, 0], image[:, :, 2] = image[:,
                                                   :, 2], image[:, :, 0].copy()
            label = cv2.imread(name.replace('image', 'label'),
                               cv2.IMREAD_UNCHANGED)
            label = crop_zero(label, reference=image)
            label = remove_too_small(label)

            if(np.sum(label) == 0):
                print('Skip empty')
                continue
            # label[label>0]=1
            image = crop_zero(image)
            fg_seg_pairs += [[image, label]]
        except Exception as e:
            print(e)
            # raise e
            pass
    print(f'{len(fg_seg_pairs)} pairs of foreground loaded.')
    return fg_seg_pairs


fg_seg_pairs = load_foreground()

ia.seed(1)


def sometimes(aug): return iaa.Sometimes(0.5, aug)


seq = iaa.Sequential(
    [
        iaa.Fliplr(0.5),  # horizontally flip 50% of all images
        iaa.SomeOf((0, 7),
                   [
            iaa.OneOf([
                iaa.GaussianBlur((0, 0.3)),
            ]),
            # These are less relevant augmentations, which may or may not help performance
            #iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
            # iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
            # iaa.AdditiveGaussianNoise(
            # loc=0, scale=(0.0, 0.02*255), per_channel=0.5
            # ),
            # iaa.Add((-15, 15), per_channel=0.5),
            # iaa.Multiply((0.8, 1.2), per_channel=0.5),
            # iaa.imgcorruptlike.Contrast(severity=1),
            # iaa.imgcorruptlike.Brightness(severity=2),
            iaa.ContrastNormalization((0.1, 1.5), per_channel=0.5),
            iaa.WithHueAndSaturation([
                iaa.WithChannels(0, iaa.Add((-15, 15))),
                iaa.WithChannels(1, iaa.Add((-20, 20))),
            ]),
            iaa.GammaContrast((0.3, 1.5)),
            iaa.WithBrightnessChannels(iaa.Add((-30, 70))),
            iaa.ScaleX((0.5, 1.5)),
            iaa.ScaleY((0.5, 1.5)),
            iaa.ShearX((-10, 10)),
            iaa.ShearY((-10, 10)),
        ],
            random_order=True
        )
    ],
    random_order=True
)


def augment_pair(fg, label):
    label_i, segmaps_aug_i = seq(images=fg, segmentation_maps=label)
    return label_i, segmaps_aug_i


#res = [1080, 960]
res = [int(1080/5), int(960/5)] # Resolution must be multiple of [9,8]
print('Resolution: ', res)
# size ratio range, numbers, blur, shear, explosure
para_space_range = {'size': [0.4, 0.8],
                    'min_num': [1, 1], 'min_area': [0.001, 0.002]}


def get_para():
    para = para_space_range.copy()
    para['min_num'] = np.random.uniform(
        low=para_space_range['min_num'][0], high=para_space_range['min_num'][1])
    para['size'] = np.random.uniform(low=para_space_range['size'][0], high=para_space_range['size'][1],
                                     size=int(para['min_num'])*20)
    para['min_area'] = np.random.uniform(
        low=para_space_range['min_area'][0], high=para_space_range['min_area'][1])
    return para


def get_pair_PIL():
    idx = np.random.randint(0, fg_seg_pairs.__len__())
    fg_pair = fg_seg_pairs[idx]
    fg = fg_pair[0].copy()
    fg = Image.fromarray(fg, 'RGBA')
    label = Image.fromarray(fg_pair[1], 'RGBA')
    return fg, label


# Background File Buffer
print(f'Scanning for backgrounds in {bg_paths}')
bg_files = []
for bg_path in bg_paths:
    bg_files += list(filter(lambda f: True or os.path.isfile(f),
                            glob.glob(bg_path, recursive=False)))
print('{} backgrounds found. '.format(bg_files.__len__()))
buffer = []
buffered_files = []


def get_bg_pair(do_buffer=True):
    files = bg_files
    idx = np.random.randint(0, bg_path.__len__())
    if(do_buffer and files[idx] in buffered_files):
        idx = np.random.randint(0, buffer.__len__())
        bg = buffer[idx][0]
        bg_label = buffer[idx][1]
    else:
        bg = Image.open(files[idx])
        bg = bg.resize(res)
        bg_label = Image.new('RGBA', bg.size, (0, 0, 0, 0))
        buffer.append((bg, bg_label))
        buffered_files.append(files[idx])
    return bg.copy(), bg_label.copy()


def area_percent(img):
    img_full = img.copy()
    img_full[:] = 255
    return np.sum(img)/np.sum(img_full)


def get_blended(plot=False, augment=True):
    bg, bg_label = get_bg_pair()  # an image and an empty label of size res
    para = get_para()
    n = 0
    while area_percent(np.array(bg_label)) < para['min_area'] and n < np.min(para['min_num']):
        fg, label = get_pair_PIL()
        newsize = (np.array(res)*para['size'][n %
                                              int(para['size'].shape[0])]).astype(np.int)
        fg = fg.resize(newsize)
        label = label.resize(newsize)

        xoff, yoff = (np.random.rand()-0.5)*50, (np.random.rand()-0.5)*50
        loc = (int(res[0]/2-newsize[0]/2+xoff),
               int(res[1]/2-newsize[1]/2+yoff))

        bg.paste(fg, loc, fg)
        bg_label.paste(label, loc, label)
        n += 1

    bg, bg_label = np.array(bg), np.array(bg_label)
    bg = np.expand_dims(bg, axis=0)  # [:,:,:,0:3]
    bg_label = np.expand_dims(bg_label, axis=0)
    # Augmentation
    if augment:
        bg, bg_label = augment_pair(np.array(bg), np.array(bg_label))
    bg, bg_label = bg.squeeze(), bg_label.squeeze()
    if plot:
        plt.figure(figsize=(5, 5))
        plt.imshow(bg.squeeze())
        plt.imshow(bg_label.squeeze())
    return bg, bg_label
