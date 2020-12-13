NUM_CHANNELS = 3  # RGB images
PIXEL_DEPTH = 255
NUM_LABELS = 2
TEST_SIZE = 50
TRAINING_SIZE = 100
VALIDATION_SIZE = 0  # Size of the validation set.
SEED = None  # Set to None for random seed.
BATCH_SIZE = 16  # 64
NUM_EPOCHS = 300
RESTORE_MODEL = True  # If True, restore existing model instead of training a new one
RECORDING_STEP = 0
PIXEL_THRESHOLD = 0.25
DROPOUT_PROBABILITY = 0.0
# Set image patch size in pixels
# IMG_PATCH_SIZE should be a multiple of 4
# image size should be an integer multiple of this number!
IMG_PATCH_SIZE = 16
TRAIN_DIR = 'data/training/'
TEST_DIR = 'data/test/'
SAVE_NETWORK_FILE = 'networks/unet.h5'
SAVE_NETWORK_FILE_TRAINING = 'networks/unet_training.h5'
OUTPUT_DIR = 'data/predictions/'