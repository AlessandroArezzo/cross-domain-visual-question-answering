import argparse
import json
import urllib
import urllib.request
from urllib.error import HTTPError
import os

parser = argparse.ArgumentParser()
parser.add_argument('--json_path', type=str, default='', help='path of the json file')
parser.add_argument('--output_path', type=str, default='',
                    help='path to save images downloaded')
parser.add_argument('--img_url', type=str, default='img_url', help='label image url into the json object')
parser.add_argument('--img_set', type=str, default='split', help='set of the image (train, test or val)')
parser.add_argument('--ext_files', type=str, default='jpg', help='extension of the files images')
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

if __name__ == '__main__':
    sets = ["train", "test"]
    for set in sets:
        dir_image = os.path.join(opt.output_path, set)
        if not os.path.exists(dir_image):
            os.makedirs(dir_image)
    read_images_from_file()