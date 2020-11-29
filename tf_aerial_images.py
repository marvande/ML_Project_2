"""
Baseline for machine learning project on road segmentation.
This simple baseline consits of a CNN with two convolutional+pooling layers with a soft-max loss

Credits: Aurelien Lucchi, ETH Zürich

This was last tested with TensorFlow 1.13.2, which is not completely up to date.
To 'downgrade': pip install --upgrade tensorflow==1.13.2
"""

import gzip
import os
import sys
import urllib
import matplotlib.image as mpimg
from PIL import Image
import code
import tensorflow.python.platform
import numpy
import tensorflow as tf
import tensorflow.keras.layers as layers
import source.mask_to_submission as submission_maker
import source.constants as cst
import source.images as images

def error_rate(predictions, labels):
    """Return the error rate based on dense predictions and 1-hot labels."""
    return 100.0 - (
        100.0 *
        numpy.sum(numpy.argmax(predictions, 1) == numpy.argmax(labels, 1)) /
        predictions.shape[0])

# Write predictions from neural network to a file
def write_predictions_to_file(predictions, labels, filename):
    max_labels = numpy.argmax(labels, 1)
    max_predictions = numpy.argmax(predictions, 1)
    file = open(filename, "w")
    n = predictions.shape[0]
    for i in range(0, n):
        file.write(max_labels(i) + ' ' + max_predictions(i))
    file.close()

# Print predictions from neural network
def print_predictions(predictions, labels):
    max_labels = numpy.argmax(labels, 1)
    max_predictions = numpy.argmax(predictions, 1)
    print(str(max_labels) + ' ' + str(max_predictions))

# Convert array of labels to an image
def label_to_img(imgwidth, imgheight, w, h, labels):
    array_labels = numpy.zeros([imgwidth, imgheight])
    idx = 0
    for i in range(0, imgheight, h):
        for j in range(0, imgwidth, w):
            if labels[idx][0] > 0.5:  # bgrd
                l = 0
            else:
                l = 1
            array_labels[j:j+w, i:i+h] = l
            idx = idx + 1
    return array_labels


# Make an image summary for 4d tensor image with index idx
def get_image_summary(img, idx=0):
    V = tf.slice(img, (0, 0, 0, idx), (1, -1, -1, 1))
    img_w = img.get_shape().as_list()[1]
    img_h = img.get_shape().as_list()[2]
    min_value = tf.reduce_min(V)
    V = V - min_value
    max_value = tf.reduce_max(V)
    V = V / (max_value * cst.PIXEL_DEPTH)
    V = tf.reshape(V, (img_w, img_h, 1))
    V = tf.transpose(V, (2, 0, 1))
    V = tf.reshape(V, (-1, img_w, img_h, 1))
    return V

# Make an image summary for 3d tensor image with index idx
def get_image_summary_3d(img):
    V = tf.slice(img, (0, 0, 0), (1, -1, -1))
    img_w = img.get_shape().as_list()[1]
    img_h = img.get_shape().as_list()[2]
    V = tf.reshape(V, (img_w, img_h, 1))
    V = tf.transpose(V, (2, 0, 1))
    V = tf.reshape(V, (-1, img_w, img_h, 1))
    return V

# Get a concatenation of the prediction and groundtruth for given input file
def get_prediction_with_groundtruth(filename, image_idx, is_training = True):

    imageid = ""
    if is_training:
        imageid = "satImage_%.3d" % image_idx
    else:
        imageid = "test_%d/test_%d" % (image_idx, image_idx)
    image_filename = filename + imageid + ".png"
    img = mpimg.imread(image_filename)

    img_prediction = get_prediction(img)
    cimg = images.concatenate_images(img, img_prediction)

    return cimg    

def get_groundtruth(filename, image_idx, is_training = True):
    imageid = ""
    if is_training:
        imageid = "satImage_%.3d" % image_idx
    else:
        imageid = "test_%d/test_%d" % (image_idx, image_idx)
    image_filename = filename + imageid + ".png"
    img = mpimg.imread(image_filename)
    # blank = Image.new("L", (0, 0))
    return get_prediction(img)

# Get prediction overlaid on the original image for given input file
def get_prediction_with_overlay(filename, image_idx, is_training = True):
    imageid = ""
    if is_training:
        imageid = "satImage_%.3d" % image_idx
    else:
        imageid = "test_%d/test_%d" % (image_idx, image_idx)
    image_filename = filename + imageid + ".png"
    img = mpimg.imread(image_filename)

    img_prediction = get_prediction(img)
    oimg = images.make_img_overlay(img, img_prediction)

    return oimg

def get_unet():
    inputs = layers.Input((200, 200, 3), name="input_layer")
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D((2, 2), (2, 2))(conv1)
    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D((2, 2), (2, 2))(conv2)
    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D((2, 2), (2, 2))(conv3)
    conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = layers.MaxPooling2D((2, 2), (2, 2))(conv4)
    conv5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(conv5)
    up6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5)
    cropped6 = tf.image.resize_with_crop_or_pad(conv4, up6.shape[1], up6.shape[2])
    conc6 = layers.concatenate([up6, cropped6], axis=3)
    conv6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conc6)
    conv6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)
    up7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6)
    cropped7 = tf.image.resize_with_crop_or_pad(conv3, up7.shape[1], up7.shape[2])
    conc7 = layers.concatenate([up7, cropped7], axis=3)
    conv7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conc7)
    conv7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)
    up8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7)
    cropped8 = tf.image.resize_with_crop_or_pad(conv2, up8.shape[1], up8.shape[2])
    conc8 = layers.concatenate([up8, cropped8], axis=3)
    conv8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conc8)
    conv8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)
    up9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
    cropped9 = tf.image.resize_with_crop_or_pad(conv1, up9.shape[1], up9.shape[2])
    conc9 = layers.concatenate([up9, cropped9], axis=3)
    conv9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conc9)
    conv10 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)
    conv10 = layers.Conv2D(1, (1, 1), activation='sigmoid', name="output_layer")(conv10)

    unet = tf.keras.Model(inputs=[inputs], outputs=[conv10])
    unet.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return unet


def train_unet(unet):
    
    train_data_filename = cst.TRAIN_DIR + 'images/'
    train_labels_filename = cst.TRAIN_DIR + 'groundtruth/' 

    # Extract it into numpy arrays.
    train_data = images.load_training(train_data_filename, cst.TRAINING_SIZE)
    train_labels = images.load_groundtruths(train_labels_filename, cst.TRAINING_SIZE)

    print("DATA SHAPE " + str(train_data.shape))
    print("TRAIN_LABELS SHAPE " + str(train_labels.shape))

    unet.summary()
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(cst.SAVE_NETWORK_FILE, monitor='loss', save_best_only=True)

    output_shape = unet.get_layer("output_layer").output_shape
    margin = int((train_labels.shape[1] - output_shape[1]) / 2)
    right_margin = int(output_shape[1] + margin)
    train_labels = train_labels[:,margin:right_margin,margin:right_margin]

    unet.fit(train_data, train_labels, epochs=cst.NUM_EPOCHS, validation_split=0.2, batch_size=cst.BATCH_SIZE, callbacks=[model_checkpoint])
    unet.save(cst.SAVE_NETWORK_FILE)
    train_data = None
    train_labels = None

def predict(train_before = False):
    unet = get_unet()
    if train_before or not os.path.exists(cst.SAVE_NETWORK_FILE):
        train_unet(unet)
    else:
        print("LOADING SAVED WEIGHTS")
        unet.load_weights(cst.SAVE_NETWORK_FILE)

    input_size = unet.get_layer("input_layer").input_shape[0][1]
    output_size = unet.get_layer("output_layer").output_shape[1]
    test_data = images.load_test(cst.TEST_DIR, cst.TEST_SIZE, input_size, output_size)
    
    masks = unet.predict(test_data, verbose=1)
    numpy.save("image_mask.npy", masks)

    return masks

def generate_masks(masks):
    predictions = []
    if not os.path.isdir(cst.OUTPUT_DIR):
        os.mkdir(cst.OUTPUT_DIR)
    print(masks.shape)
    for i in range(0, 800, 16):
        mask_line_1 = numpy.concatenate((masks[i], masks[i + 1], masks[i + 2], masks[i + 3]), axis=1)
        mask_line_2 = numpy.concatenate((masks[i + 4], masks[i + 5], masks[i + 6], masks[i + 7]), axis=1)
        mask_line_3 = numpy.concatenate((masks[i + 8], masks[i + 9], masks[i + 10], masks[i + 11]), axis=1)
        mask_line_4 = numpy.concatenate((masks[i + 12], masks[i + 13], masks[i + 14], masks[i + 15]), axis=1)
        mask = numpy.concatenate((mask_line_1, mask_line_2, mask_line_3, mask_line_4), axis=0)[0:608, 0:608, :]
        mask = mask.reshape((608, 608))
        mask = numpy.around(mask).astype('float32')
        for k in range(0, 608, 16):
            for l in range(0, 608, 16):
                patch = mask[k:k + 16, l:l + 16]
                summed = numpy.sum(patch)
                if summed >= (16 * 16 * cst.PIXEL_THRESHOLD):
                    mask[k:k + 16, l:l + 16].fill(1)
                else:
                    mask[k:k + 16, l:l + 16].fill(0)
        predictions.append(mask)
        Image.fromarray(images.img_float_to_uint8(mask)).save(cst.OUTPUT_DIR + "mask_%d.png" % ((i / 16) + 1))

if __name__ == '__main__':
    generate_masks(predict())

