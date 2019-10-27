import os
import time
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2
import shutil


foregound_path = './foreground/*.png'
background_path = './background/*.png'
fg_files = glob.glob(foregound_path)
bg_files = glob.glob(background_path)
for name in bg_files:
    print(name)
    image = plt.imread(name)
