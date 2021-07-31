import os

# dataset image directores
TRAIN_IMAGE = "../datasets/tiny-imagenet-200/train/"
TEST_IMAGE = "../datasets/tiny-imagenet-200/test/"
VAL_IMAGE = "../datasets/tiny-imagenet-200/val/images"

# label annotations, mappings
WORD_MAPPING = "../datasets/tiny-imagenet-200/words.txt"
VAL_MAPPING = "../datasets/tiny-imagenet-200/val/val_annotations.txt"

# HDF5 dataset directories
HDF_PATH = "../datasets/hdf5"
os.makedirs(HDF_PATH, exist_ok=True)

TRAIN_HDF5 = os.path.sep.join([HDF_PATH, "train.hdf5"])
TEST_HDF5 = os.path.sep.join([HDF_PATH, "test.hdf5"])
VAL_HDF5 = os.path.sep.join([HDF_PATH, "val.hdf5"])

EPOCHS = 100
BATCH_SIZE = 1000
DECAY_POWER = 1
LEARNING_RATE = 1e-03

NUM_CLASSES = 200
NUM_IMAGES = 500 * NUM_CLASSES
TEST_SIZE = int(0.1 * NUM_IMAGES)

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64

# output paths
OUTPUT = "../outputs"
os.makedirs(OUTPUT, exist_ok=True)

# path to store dataset's RGB mean and trained model
DATASET_MEAN = os.path.sep.join([OUTPUT, "mean.json"])
MODEL_PATH = os.path.sep.join([OUTPUT, "deepergooglenet-tiny-imagenet.h5"])

# path to store model metrics during train
FIG_PATH = os.path.sep.join([OUTPUT, "tiny-imagenet.png"])
JSON_PATH = os.path.sep.join([OUTPUT, "tiny-imagenet.json"])

# extract labels from image path, mappings
def get_label(path):
    label = path.split(os.path.sep)[-3]
    return label

# calculates generator steps
def cal_steps(n_samples, batch_size):
    steps = n_samples // batch_size
    return steps if (steps * batch_size) == n_samples else steps + 1

# weight decay callback
def poly_weight_decay(current_epoch):
    lr = LEARNING_RATE * (1 - (current_epoch/EPOCHS))**DECAY_POWER
    return lr