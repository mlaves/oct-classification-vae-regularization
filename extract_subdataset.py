import glob
from collections import OrderedDict
import random
from shutil import copy2
import os
from tqdm import tqdm


def extract_subdataset(n: int=2000, m: int=1000):
    """
    This function is used to extract a limited subdataset from the Kermany OCT dataset.

    :param n: Numer of images for training set
    :param m: Number of images for validation set
    """

    random.seed(0)

    in_path = "/home/laves/Downloads/OCT2017/train/"
    train_out_path = "/home/laves/Downloads/OCT2017_3/train/"
    val_out_path = "/home/laves/Downloads/OCT2017_3/val/"
    test_out_path = "/home/laves/Downloads/OCT2017_3/test/"

    class_paths = glob.glob(in_path + "/*")
    class_list = [c.split('/')[-1] for c in class_paths]
    img_paths = OrderedDict([
        (class_list[0], sorted(glob.glob(class_paths[0] + '/*.jpeg'))),
        (class_list[1], sorted(glob.glob(class_paths[1] + '/*.jpeg'))),
        (class_list[2], sorted(glob.glob(class_paths[2] + '/*.jpeg'))),
        (class_list[3], sorted(glob.glob(class_paths[3] + '/*.jpeg'))),
    ])

    # get n images for training
    for c in tqdm(class_list):
        # sample n images
        sample = random.sample(img_paths[c], n//len(class_list))

        # create out path
        if not os.path.exists(train_out_path + c + '/'):
            os.makedirs(train_out_path + c + '/', exist_ok=True)

        for f in sample:
            copy2(src=f, dst=train_out_path + c + '/')

        # remove extracted images
        img_paths[c] = [x for x in img_paths[c] if x not in sample]

    # get m images for validation
    for c in tqdm(class_list):
        # sample n images
        sample = random.sample(img_paths[c], m//len(class_list))

        # create out path
        if not os.path.exists(val_out_path + c + '/'):
            os.makedirs(val_out_path + c + '/', exist_ok=True)

        for f in sample:
            copy2(src=f, dst=val_out_path + c + '/')

        # remove extracted images
        img_paths[c] = [x for x in img_paths[c] if x not in sample]

    # get rest for testing
    for c in tqdm(class_list):
        # create out path
        if not os.path.exists(test_out_path + c + '/'):
            os.makedirs(test_out_path + c + '/', exist_ok=True)

        for f in img_paths[c]:
            copy2(src=f, dst=test_out_path + c + '/')


if __name__ == '__main__':
    extract_subdataset()
