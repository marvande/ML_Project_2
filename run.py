import sys
import urllib
import matplotlib.image as mpimg
from PIL import Image
import code
import tensorflow.python.platform
import numpy
from tensorflow import keras
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.backend as backend
import source.mask_to_submission as submission_maker
import source.constants as cst
import source.images as images
import os
import argparse


def recall(y, predictions):
    true_positives = backend.sum(backend.round(backend.clip(y * predictions, 0, 1)))
    possible_positives = backend.sum(backend.round(backend.clip(y, 0, 1)))
    recall = true_positives / (possible_positives + backend.epsilon())
    return recall

def precision(y, predictions):
    true_positives = backend.sum(backend.round(backend.clip(y * predictions, 0, 1)))
    predicted_positives = backend.sum(backend.round(backend.clip(predictions, 0, 1)))
    precision = true_positives / (predicted_positives + backend.epsilon())
    return precision

def f1_metric(y, predictions):
    pre = precision(y, predictions)
    rec = recall(y, predictions)
    return 2 * ((pre * rec) / (pre + rec + backend.epsilon()))

class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=1.0, gamma=2.0, reduction=tf.keras.losses.Reduction.AUTO, name='focal_loss'):
        super().__init__(reduction=reduction, name=name)
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        y_true = tf.expand_dims(y_true, axis=-1)

        bce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        pt = tf.math.exp(-bce_loss)

        return self.alpha * tf.math.pow(1-pt, self.gamma) * bce_loss -\
               (1-self.alpha) * tf.math.pow(pt, self.gamma) * tf.math.log(1-pt)

def get_unet():
    inputs = layers.Input((200, 200, 3), name="input_layer")
    drop1 = layers.Dropout(cst.DROPOUT_PROBABILITY)(inputs)
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(drop1)
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
    drop10 = layers.Dropout(cst.DROPOUT_PROBABILITY)(conv9)
    conv10 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(drop10)
    conv10 = layers.Conv2D(1, (1, 1), activation='sigmoid', name="output_layer")(conv10)

    unet = tf.keras.Model(inputs=[inputs], outputs=[conv10])
    unet.compile(optimizer='adam', loss=FocalLoss(alpha=0.75, gamma=5.0), metrics=[f1_metric, 'accuracy'])
    return unet


def train_unet(unet):
    train_data_filename = cst.TRAIN_DIR + 'images/'
    train_labels_filename = cst.TRAIN_DIR + 'groundtruth/'

    # Extract it into numpy arrays.
    train_data, mean_train, std_train = images.load_training(train_data_filename, cst.TRAINING_SIZE)
    train_labels = images.load_groundtruths(train_labels_filename, cst.TRAINING_SIZE)

    print("DATA SHAPE " + str(train_data.shape))
    print("TRAIN_LABELS SHAPE " + str(train_labels.shape))

    unet.summary()
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(cst.SAVE_NETWORK_FILE, monitor='f1_metric',
                                                          save_best_only=True)

    output_shape = unet.get_layer("output_layer").output_shape
    margin = int((train_labels.shape[1] - output_shape[1]) / 2)
    right_margin = int(output_shape[1] + margin)
    train_labels = train_labels[:, margin:right_margin, margin:right_margin]

    history = unet.fit(train_data, train_labels, epochs=cst.NUM_EPOCHS, validation_split=0.0, batch_size=cst.BATCH_SIZE,
             callbacks=[model_checkpoint])

    del train_data
    del train_labels

    return mean_train, std_train, history


def predict():
    unet = get_unet()
    mean_train, std_train, history = train_unet(unet)

    input_size = unet.get_layer("input_layer").input_shape[0][1]
    output_size = unet.get_layer("output_layer").output_shape[1]

    test_data = images.load_test(cst.TEST_DIR, cst.TEST_SIZE, input_size, output_size, mean_train, std_train)

    masks = unet.predict(test_data, verbose=1)
    numpy.save("image_mask.npy", masks)

    return masks, history

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
        mask = numpy.around(mask).astype('float64')
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



def parse_args():
        """
        Parse command line flags.
        :return results: Namespace of the arguments to pass to the main run function.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('-l', action='store_true', default=True, dest='load', help='Load trained model and predict')
        parser.add_argument('-t', action='store_true', default=False, dest='train', help='Train model from scratch')
        parser.add_argument('-s', type=str, dest='subname', default='submission.csv' ,help='submission file name')
    
        results = parser.parse_args()
    
        return results


if __name__ == '__main__':
    args = parse_args()
    if args.load:
        print("Loading model:")
        unet = keras.models.load_model('ourmodel.h5', custom_objects={'FocalLoss': FocalLoss(alpha=0.75, gamma=5.0), 'f1_metric':f1_metric})
        print("[Success] Model successfully loaded")
        train_data_filename = cst.TRAIN_DIR + 'images/'
        train_labels_filename = cst.TRAIN_DIR + 'groundtruth/' 
        input_size = unet.get_layer("input_layer").input_shape[0][1]
        output_size = unet.get_layer("output_layer").output_shape[1]
        train_data, mean_train, std_train = images.load_training(train_data_filename, cst.TRAINING_SIZE)
        test_data = images.load_test(cst.TEST_DIR, cst.TEST_SIZE, input_size, output_size, mean_train, std_train)
        print("Predicting from test images:")
        masks = unet.predict(test_data, verbose=1)
        numpy.save("image_mask.npy", masks)
        generate_masks(masks)
        print("[Success] Predictions done")

    else:
        masks, history = predict()
        generate_masks(masks)

    submission_filename = args.subname
    image_filenames = []
    
    print("Creating Submission file")
    for i in range(1, 51):
        image_filename = 'data/predictions/mask_' + '%d' % i + '.png'
        print(image_filename)
        image_filenames.append(image_filename)
    submission_maker.masks_to_submission(submission_filename, *image_filenames)
    print("[Success] Submission file successfully created")
    