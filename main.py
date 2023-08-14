import os
import cv2
import json
import torch
import h5py
import pyarrow as pa
import pyarrow.parquet as pq
import lmdb
import numpy as np
import pandas as pd
import torch.nn as nn
import skimage.io as io
import webdataset as wds
import matplotlib.pyplot as plt

from time import time
from hashlib import sha512
from pycocotools.coco import COCO

from tqdm import tqdm
os.environ["WDS_VERBOSE_CACHE"] = "1"

COCO_PATH = "/Users/sungjun/datasets/coco"

def draw_box(img, bbox):
    x1, y1 = bbox[0], bbox[1]
    x2, y2 = x1 + bbox[2], y1 + bbox[3]
    img = cv2.rectangle(np.array(img), list(map(int, [x1, y1])), list(map(int, [x2, y2])), (255,255,0))
    cv2.imwrite("hi.png", img)


def main():
    image_path = os.path.join(COCO_PATH, "images")
    anno_path = os.path.join(COCO_PATH, "annotations")

    im_dict = {}
    anno_dict = {}
    for dtype in ["train2017", "val2017"]:
        im_dict[dtype] = {'checksum': [], 'image_id' : [], 'image': [], 'shape': []}
        anno_dict[dtype] ={'image_id': [], 'category_id': [], 'bbox': [], 'image_shape' : []}

        coco = COCO(os.path.join(anno_path, f"instances_{dtype}.json"))
        imgIds = coco.getImgIds()
        print("Total images: {}".format(len(imgIds)))
        
        
        if "train" in dtype:
            cats = coco.loadCats(coco.getCatIds())
            print("Number of categories: {}".format(len(cats)))
            nms=[cat['name'] for cat in cats]
            print('\nCOCO categories: \n{}\n'.format(' '.join(nms)))
        
        for img_info in tqdm(coco.loadImgs(imgIds)):
            im_name = img_info['coco_url'].split('/')[-1]
            im_path = os.path.join(COCO_PATH, "images", dtype, im_name)
            im = cv2.imread(im_path)

            if len(im.shape) == 2:
                np.repeat(im[..., np.newaxis], 3, -1)
            
            anno_id = coco.getAnnIds(imgIds=img_info['id'])
            annos = coco.loadAnns(anno_id)
            
            #coco.showAnns(annos)
            #plt.imshow(im); plt.axis('off')
            start_time = time()
            
            for ann in annos:
                anno_dict[dtype]['image_id'].append(ann['image_id'])
                anno_dict[dtype]['bbox'].append(ann['bbox'])
                anno_dict[dtype]['category_id'].append(ann['category_id'])
                anno_dict[dtype]['image_shape'].append(im.shape)

            im_dict[dtype]['shape'].append(im.shape)
            im_dict[dtype]['image_id'].append(ann['image_id'])
            
            im = np.ravel(im).tobytes() #np.ndarray to byte with flatten
            im_dict[dtype]['image'].append(im)
            
            
        im_df = pd.DataFrame(im_dict[dtype])
        table = pa.Table.from_pandas(im_df)
        output_file = f'{dtype}_image.parquet'
        pq.write_table(table, output_file)

        

            

if __name__ == "__main__":
    main()