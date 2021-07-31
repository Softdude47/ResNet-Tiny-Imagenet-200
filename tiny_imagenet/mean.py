import cv2
import sys
import json
import numpy as np
import progressbar
from imutils.paths import list_images

sys.path.append("../")
from tiny_imagenet.configs import tiny_imagenet_config as config


# gets paths of train images
paths= list(list_images(config.TRAIN_IMAGE))

# initialize lists to store RGB mean values of each train images
B, G, R = [], [], []

# constructs progressbar widget
widgets = ["[INFO]: Building Dataset: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]

# start progressbar
pbar = progressbar.ProgressBar(max_value=len(paths), widgets=widgets).start()

# loop over images paths in batches
for i in np.arange(0, len(paths), config.BATCH_SIZE):
    
    # select current batch of image paths
    batch_path = paths[i : i + config.BATCH_SIZE]
    
    # load images from current batch of image paths
    batch_images = [cv2.imread(p) for p in batch_path]
    batch_images = [cv2.resize(image, (config.IMAGE_WIDTH, config.IMAGE_HEIGHT)) for image in batch_images]
    
    # extract mean RGB values from each image in currenct image batch
    for image in batch_images:
        b, g, r = cv2.mean(image.astype("float"))[:3]
        B.append(b)
        G.append(g)
        R.append(r)
    
    # update progressbar after each batches
    pbar.update(config.BATCH_SIZE)
    
# stops progressbar
pbar.finish()

# average the RGB mean values 
mean = {"B": np.mean(B), "G": np.mean(G), "R": np.mean(R)}

# save the averaged mean values to json file
with open(config.DATASET_MEAN, "w") as f:
    f.write(json.dumps(mean))
    f.close()

print("[INFO] RGB mean values saved")