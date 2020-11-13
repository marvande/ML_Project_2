import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
import os,sys
from PIL import Image

def plot_double(arr1, arr2, title1, title2):
    figure, ax = plt.subplots(1, 2, figsize=(20, 20))
    ax[0].imshow(arr1, cmap='gray')
    ax[0].set_title(title1)
    ax[1].imshow(arr2, cmap='gray')
    ax[1].set_title(title2)