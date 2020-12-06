import gzip
import os
import sys
import urllib
import matplotlib.image as mpimg
from PIL import Image
import code
import numpy
import source.constants as cst
import scipy.ndimage as scp

def img_float_to_uint8(img):
    rimg = img - numpy.min(img)
    rimg = (rimg / numpy.max(rimg) * cst.PIXEL_DEPTH).round().astype(numpy.uint8)
    return rimg

# Assign a label to a patch v
def value_to_class(v):
    foreground_threshold = 0.25  # percentage of pixels > 1 required to assign a foreground label to a patch
    df = numpy.sum(v)
    if df > foreground_threshold:  # road
        return [0, 1]
    else:  # bgrd
        return [1, 0]

def concatenate_images(img, gt_img):
    n_channels = len(gt_img.shape)
    w = gt_img.shape[0]
    h = gt_img.shape[1]
    if n_channels == 3:
        cimg = numpy.concatenate((img, gt_img), axis=1)
    else:
        gt_img_3c = numpy.zeros((w, h, 3), dtype=numpy.uint8)
        gt_img8 = img_float_to_uint8(gt_img)          
        gt_img_3c[:, :, 0] = gt_img8
        gt_img_3c[:, :, 1] = gt_img8
        gt_img_3c[:, :, 2] = gt_img8
        img8 = img_float_to_uint8(img)
        cimg = numpy.concatenate((img8, gt_img_3c), axis=1)
    return cimg

def make_img_overlay(img, predicted_img):
    w = img.shape[0]
    h = img.shape[1]
    color_mask = numpy.zeros((w, h, 3), dtype=numpy.uint8)
    color_mask[:, :, 0] = predicted_img*cst.PIXEL_DEPTH

    img8 = img_float_to_uint8(img)
    background = Image.fromarray(img8, 'RGB').convert("RGBA")
    overlay = Image.fromarray(color_mask, 'RGB').convert("RGBA")
    new_img = Image.blend(background, overlay, 0.2)
    return new_img

    # Extract patches from a given image
def img_crop(im, w, h):
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches

def load_training(filename, num_images):
    imgs = []
    for i in range(1, num_images+1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print('File ' + image_filename + ' does not exist')
    data = numpy.asarray(imgs).astype('float64')
    mean = numpy.mean(data)
    std = numpy.std(data)
    data -= mean
    data /= std
    result = numpy.concatenate((data[:, 0:200, 0:200, :], data[:, 0:200, 200:400, :]))
    result = numpy.concatenate((result, data[:, 200:400, 0:200, :]))
    result = numpy.concatenate((result, data[:, 200:400, 200:400, :]))
    rotated = numpy.rot90(result, axes=(1, 2))
    result = numpy.concatenate((result, rotated))
    # rotated = numpy.rot90(rotated, axes=(1, 2))
    # # result = numpy.concatenate((result, rotated))
    # rotated = numpy.rot90(rotated, axes=(1, 2))
    # result = numpy.concatenate((result, rotated))

    rotated = scp.rotate(data, 45, axes=(1, 2))
    numpy.set_printoptions(threshold=sys.maxsize)
    result = numpy.concatenate((result, rotated[:, 183:383, 183:383, :]))
    rotated = scp.rotate(data, 135, axes=(1, 2))
    result = numpy.concatenate((result, rotated[:, 183:383, 183:383, :]))
    rotated = scp.rotate(data, 225, axes=(1, 2))
    result = numpy.concatenate((result, rotated[:, 183:383, 183:383, :]))
    rotated = scp.rotate(data, 315, axes=(1, 2))
    result = numpy.concatenate((result, rotated[:, 183:383, 183:383, :]))

    print("DATA SHAPE ", result.shape)

    return result

def load_groundtruths(filename, num_images):
    gt_imgs = []
    for i in range(1, num_images + 1):
        imageid = "satImage_%.3d" % i
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            gt_imgs.append(img)
        else:
            print('File ' + image_filename + ' does not exist')
    labels = numpy.asarray(gt_imgs)
    labels = numpy.around(labels)
    result = numpy.concatenate((labels[:, 0:200, 0:200], labels[:, 0:200, 200:400]))
    result = numpy.concatenate((result, labels[:, 200:400, 0:200]))
    result = numpy.concatenate((result, labels[:, 200:400, 200:400]))
    rotated = numpy.rot90(result, axes=(1, 2))
    result = numpy.concatenate((result, rotated))
    # rotated = numpy.rot90(rotated, axes=(1, 2))
    # # result = numpy.concatenate((result, rotated))
    # rotated = numpy.rot90(rotated, axes=(1, 2))
    # result = numpy.concatenate((result, rotated))

    rotated = scp.rotate(labels, 45, axes=(1, 2))
    # print(numpy.abs(numpy.around(rotated[0, 183:383, 183:383])))
    result = numpy.concatenate((result, numpy.abs(numpy.around(rotated[:, 183:383, 183:383]))))
    rotated = scp.rotate(labels, 135, axes=(1, 2))
    result = numpy.concatenate((result, numpy.abs(numpy.around(rotated[:, 183:383, 183:383]))))
    rotated = scp.rotate(labels, 225, axes=(1, 2))
    result = numpy.concatenate((result, numpy.abs(numpy.around(rotated[:, 183:383, 183:383]))))
    rotated = scp.rotate(labels, 315, axes=(1, 2))
    result = numpy.concatenate((result, numpy.abs(numpy.around(rotated[:, 183:383, 183:383]))))
    print("LABEL SHAPE ", result.shape)

    return result.astype('float64')

def load_test(filename, num_images, input_size, output_size):
    imgs = []
    for i in range(1, num_images + 1):
        imageid = "test_%d/test_%d" % (i, i)
        image_filename = filename + imageid + ".png"
        if os.path.isfile(image_filename):
            print('Loading ' + image_filename)
            img = mpimg.imread(image_filename)
            imgs.append(img)
        else:
            print('File ' + image_filename + ' does not exist')
    data = numpy.asarray(imgs).astype('float64')
    mean = numpy.mean(data)
    std = numpy.std(data)
    data -= mean
    data /= std

    expanded = []
    original_size = data.shape[1]

    for k in range(num_images):
        img = data[k]
        hflip = numpy.fliplr(img)
        vflip = numpy.flipud(img)
        hvflip = numpy.flipud(hflip)
        flipped_border_line = numpy.concatenate((hvflip, vflip, hvflip), axis = 1)
        flipped_middle_line = numpy.concatenate((hflip, img, hflip), axis = 1)
        all_flipped = numpy.concatenate((flipped_border_line, flipped_middle_line, flipped_border_line))
        for i in range(0, original_size, output_size):
            for j in range(0, original_size, output_size):
                starti = original_size + i - int((input_size - output_size) / 2)
                endi = starti + input_size
                startj = original_size + j - int((input_size - output_size) / 2)
                endj = startj + input_size
                patch = all_flipped[starti:endi, startj:endj, :]
                expanded.append(patch[..., numpy.newaxis])

    return numpy.asarray(expanded)