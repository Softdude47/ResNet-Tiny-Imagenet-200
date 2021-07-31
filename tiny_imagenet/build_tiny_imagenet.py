import os
import sys
import cv2
import json
import numpy as np
import progressbar
from imutils.paths import list_images
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

sys.path.append("../")
from mas_lib.io.hdf5datasetwriter import HDF5DatasetWriter
from tiny_imagenet.configs import tiny_imagenet_config as config

# load dataset
train_path = list(list_images(config.TRAIN_IMAGE))
train_labels = [config.get_label(path) for path in train_path]

# encode dataset labels
le = LabelBinarizer()
train_labels = le.fit_transform(train_labels)

# split dataset to train and test set
(train_path, test_path, train_labels, test_labels) = train_test_split(train_path, train_labels, test_size=config.TEST_SIZE, stratify=train_labels)

# load validation dataset mappings
mappings = open(config.VAL_MAPPING).read().split("\n")
mappings = [line.split("\t")[:2] for line in mappings]

# get validation image path and labels from mapping
val_path = [os.path.sep.join([config.VAL_IMAGE, m[0]]) for m in mappings]
val_labels = [m[1] for m in mappings]

# dataset splits
datasets = (
    ("train", train_path, train_labels, config.TRAIN_HDF5),
    ("test", test_path, test_labels, config.TEST_HDF5),
    ("val", val_path, val_labels, config.VAL_HDF5),
)

# list of training dataset image mean
(R, G, B) = ([], [], [])

# loop over dataset split
for (dataset, paths, labels, hdf5_path) in datasets:
    
    # progress bar
    print(f"[INFO] Building {hdf5_path}...")
    widgets = ["[INFO] Building Dataset: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]
    pbar = progressbar.ProgressBar(max_value=len(paths), widgets=widgets).start()
    
    # initialize database
    writer = HDF5DatasetWriter(hdf5_path, "images", 1000, (len(paths), config.IMAGE_WIDTH, config.IMAGE_HEIGHT, 3))
    writer.store_class_label(le.classes_)
    
    # loop over dataset split's image path and corresponding label
    for (path, label) in zip(paths, labels):
        # load image from path
        image = cv2.imread(path)
        image = cv2.cvtColor(image.astype("float"), cv2.COLOR_BGR2RGB)
        
        # calculate mean of train split image channels
        if datasets == "train":
            (r, g, b) = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)
            
        # adds image and its label to database
        # and update progress bar
        writer.add([image,], [label,])
        pbar.update(1)
        
    # closes database and finishes progressbar
    writer.close()
    pbar.finish()

# serializes the RGB mean to JSON file
mean = {"R": np.mean(R), "G": np.mean(G), "B": np.mean(B)}
with open(config.DATASET_MEAN, "w") as f:
    f.write(json.dumps(mean))