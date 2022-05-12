from enum import EnumMeta
import os 
import sys 

import json 
from scipy.io import loadmat
import cv2
from tqdm import tqdm, trange



def conver_ps_2_coco(root_path):
    training_path = os.path.join(root_path, "training")
    val_path = os.path.join(root_path, "val")

    # convert training path
    annots = []

    anns_json = []
    imgs_json = []


    ann_idx = 0
    img_idx = 0
    
    for idx_file, file_name in enumerate(tqdm(os.listdir(training_path))):
        if file_name.split('.')[-1] == 'mat':
            annot = loadmat(os.path.join(training_path, file_name))
            for idx_ann, slot in enumerate(annot['slots']):
                p_0 = slot[0]
                p_1 = slot[1]
                ps_points = [int(annot['marks'][p_0-1][0]), int(annot['marks'][p_0-1][1]),
                             int(annot['marks'][p_1-1][0]), int(annot['marks'][p_1-1][1])]
                cat = slot[2]
                angle = slot[3]
            
                ann_json = {
                    "id": ann_idx,
                    "image_id": img_idx,
                    "category_id": int(cat),
                    "ps_points": ps_points,
                    "angles": [int(angle)]
                }

                img = cv2.imread(os.path.join(training_path, file_name.split('.')[0] + '.jpg'))
                size = img.shape
                img_json = {
                    "id": img_idx,
                    "file_name": file_name.split('.')[0] + '.jpg',
                    "width": size[1],
                    "height": size[0]
                }

                ann_idx += 1
            img_idx += 1

            anns_json.append(ann_json)
            imgs_json.append(img_json)

    save_json("traing.json", imgs_json=imgs_json, anns_json=anns_json, )

def save_json(target, imgs_json, anns_json, cats=None, type="training"):
    info = {}
    licenses = []
    if cats == None:
        categories = [
            {
                "id": 1,
                "name": "1",
                "supercategory": "None"
            },
            {
                "id": 2,
                "name": "2",
                "supercategory": "None"
            },
            {
                "id": 3,
                "name": "3",
                "supercategory": "None"
            }
        ]

    training_json = {
        "info": info,
        "licenses": licenses,
        "images": imgs_json,
        "annotations": anns_json,
        "categories": categories
    }

    with open(target, 'w') as js_w:
        json.dump(training_json, js_w)


def main():
    root_path = "/home/pc/data/ps2"
    conver_ps_2_coco(root_path)

if __name__=='__main__':
    main()