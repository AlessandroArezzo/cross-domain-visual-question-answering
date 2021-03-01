import argparse
import glob
import json
import urllib
import urllib.request
from urllib.error import HTTPError
import os
from random import shuffle

from PIL import Image, UnidentifiedImageError

Image.MAX_IMAGE_PIXELS = 2553880864

parser = argparse.ArgumentParser()
parser.add_argument('--json_path', type=str, default='', help='path of the json file')
parser.add_argument('--output_path', type=str, default='',
                    help='path to save images downloaded')
parser.add_argument('--img_url', type=str, default='img_url', help='label image url into the json object')
parser.add_argument('--img_set', type=str, default='split', help='set of the image (train, test or val)')
parser.add_argument('--ext_files', type=str, default='jpg', help='extension of the files images')
parser.add_argument('--preprocessing_type', type=int, default=0, help='type of the preprocessing '
                                            '(0 for download form the json file / 1 for clear images of 0 byte size)')
parser.add_argument('--images_path', type=str, default='', help='path of the images to clear')
parser.add_argument('--n_images_to_select', type=int, default=1000, help='number of images that must be selected')


opt = parser.parse_args()

def read_images_from_file():
    with open(opt.json_path) as f:
        json_data = json.load(f)
    num_images = len(json_data)
    for (idx, image_number) in enumerate(json_data):
        print("Download images "+str(idx)+"/"+str(num_images))
        image = json_data[image_number]
        image_url = image["img_url"]
        image_set = image["split"]
        if image_set == 'val':
            image_set = 'train'
        dir_image = os.path.join(opt.output_path, image_set)
        try:
            path_image = os.path.join(dir_image, str(image_number) + "." + opt.ext_files)
            urllib.request.urlretrieve(image_url, path_image)
        except HTTPError:
            print("Image " + image["title"] + " not found")

def clear_images():
    dir = opt.images_path
    assert os.path.isdir(dir)
    files = glob.glob(os.path.join(dir) + '/*.*')
    for file in files:
        try:
            Image.open(file)
        except UnidentifiedImageError:
            os.remove(file)
            print("REMOVED IMAGE "+ str(file))

def select_images():
    dirs = opt.images_path.split(",")
    files = []
    count_imgs = 0
    for dir in dirs:
        files += glob.glob(os.path.join(dir) + '/*.*')
    shuffle(files)
    for image in files:
        try:
            img = Image.open(image)
            img.save(opt.output_path+"/"+str(count_imgs)+'.jpeg')
            count_imgs += 1
            print("Images #"+str(count_imgs)+" extracted")
            if count_imgs == opt.n_images_to_select:
                break
        except UnidentifiedImageError:
            continue
        except OSError:
            continue

if __name__ == '__main__':

    if opt.preprocessing_type == 0:
        sets = ["train", "test"]
        for set in sets:
            dir_image = os.path.join(opt.output_path, set)
            if not os.path.exists(dir_image):
                os.makedirs(dir_image)
        read_images_from_file()
    elif opt.preprocessing_type == 1:
        clear_images()
    elif opt.preprocessing_type == 2:
        select_images()