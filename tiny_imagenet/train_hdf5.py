import sys
import json
import argparse
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
# from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.preprocessing.image import ImageDataGenerator

sys.path.append("../")
from mas_lib.nn.conv.resnet import ResNet
from mas_lib.callbacks.epochcheckpoint import EpochCheckpoint
from mas_lib.callbacks.trainingmonitor import TrainingMonitor
from mas_lib.io.hdf5datasetgenerator import HDF5DatasetGenerator
from tiny_imagenet.configs import tiny_imagenet_config as config
from mas_lib.preprocessing.meanpreprocessor import MeanPreprocessor
from mas_lib.preprocessing.patchpreprocessor import PatchPreprocessor
from mas_lib.preprocessing.simplepreprocessor import SimplePreprocessor
from mas_lib.preprocessing.imagetoarraypreprocessor import ImageToArrayPreprocessor

# command line argument
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", help="path to load trained/existing model")
ap.add_argument("-start", "--start-at", help="epoch at which model training resumes", type=int)
ap.add_argument(
    "-c",
    "--checkpoint",
    required=True,
    help="path to store model checkpoints"
)
args = vars(ap.parse_args())

# load datasets mean preprocessor
mean = json.loads(open(config.DATASET_MEAN).read())

# initialize dataset preprocessor
iap = ImageToArrayPreprocessor()
pp = PatchPreprocessor(config.IMAGE_WIDTH, config.IMAGE_HEIGHT)
mp = MeanPreprocessor(rMean=mean["R"], gMean=mean["G"], bMean=mean["B"])
sp = SimplePreprocessor(config.IMAGE_WIDTH, config.IMAGE_HEIGHT)

aug = ImageDataGenerator(
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    shear_range=0.1,
    rotation_range=18,
    horizontal_flip=True
)

# initialize dataset generators
train_gen = HDF5DatasetGenerator(config.TRAIN_HDF5, "images", config.BATCH_SIZE, [pp, mp, iap], aug, classes=config.NUM_CLASSES)
test_gen = HDF5DatasetGenerator(config.TEST_HDF5, "images", config.BATCH_SIZE, [sp, mp, iap], classes=config.NUM_CLASSES)

# create model callbacks
callbacks=[
    # LearningRateScheduler(config.poly_weight_decay),
    TrainingMonitor(config.FIG_PATH, config.JSON_PATH, args.get("start_at", 0)),
    EpochCheckpoint(path=args["checkpoint"], interval=5)
]

# creates new model if there wasn't a pevious one
if args["model"] is None:
    # initialize and tune model optimizer
    opt = SGD(config.LEARNING_RATE, momentum=0.9)

    # build and compile model
    model = ResNet.build(config.IMAGE_HEIGHT, config.IMAGE_WIDTH, 3, config.NUM_CLASSES, (9, 9, 9, 2), (64, 128, 256, 512), reg=5e-4)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

# loads previously saved model(if any)
else:
    # load model
    model = load_model(args["model"])
    prev_lr = K.get_value(model.optimizer.lr)
    
    # change learning rate of model optimizer
    print(f"[INFO] Previous learning rate {prev_lr}")
    K.set_value(model.optimizer.lr, 0.00005)
    print(f"[INFO] New learning rate set to {K.get_value(model.optimizer.lr)}")

# train model
model.fit(
    train_gen,
    validation_data = test_gen.generate(),
    steps_per_epoch=config.cal_steps(train_gen.num_images, config.BATCH_SIZE),
    validation_steps = config.cal_steps(test_gen.num_image, config.BATCH_SIZE),
    callbacks=callbacks,
    epochs=config.EPOCHS
)
