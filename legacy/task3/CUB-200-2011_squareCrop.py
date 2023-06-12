from PIL import Image
import os
import xml.etree.ElementTree as ET
import scipy.io
import pickle
from pathlib import Path

def adjust(x1, y1, x2, y2):
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1
    size = max(width, height)
    new_x1 = center_x - size / 2
    new_y1 = center_y - size / 2
    new_x2 = center_x + size / 2
    new_y2 = center_y + size / 2
    return new_x1, new_y1, new_x2, new_y2

root_dir = "/mnt/sda/2022-0526/home/scc/zty/code/ANN/finalproject/new_test_project/data/CUB-200-2011/CUB_200_2011"
subfolders = [ f.path for f in os.scandir(os.path.join(root_dir,"images")) if f.is_dir() ]
# print(subfolders)
cnt = 0
for pat in subfolders:
    os.mkdir(os.path.join(pat[:79], "squarecroppedimages", pat[87:]))

Info = pickle.load(open("pred_result.pkl", "rb"))
for det in Info:
    im = Image.open(det['img_path']).convert("RGB")
    if len(det['pred_instances']['bboxes'].tolist()) == 0:
        x1, y1 = 0, 0
        x2, y2 = im.size
    else:
        x1, y1, x2, y2 = det['pred_instances']['bboxes'].tolist()[0]
    x1, y1, x2, y2 = adjust(x1, y1, x2, y2)
    im = im.crop([x1, y1, x2, y2])
    im.save(os.path.join(det['img_path'][:79], "squarecroppedimages", det['img_path'][87:]))



with open(os.path.join(root_dir, "images.txt"), 'r') as f:
    img_list = f.readlines()
with open(os.path.join(root_dir, "bounding_boxes.txt"), 'r') as f:
    bbox_list = f.readlines()
with open(os.path.join(root_dir, "image_class_labels.txt"), 'r') as f:
    class_list = f.readlines()

split_dict = {}
with open(os.path.join(root_dir, "train_test_split.txt"), 'r') as f:
    split_list = f.readlines()
    for line in split_list:
        img_id, is_train = line.strip().split()
        split_dict[int(img_id)] = int(is_train)

for img_line, bbox_line, class_line in zip(img_list, bbox_list, class_list):
    img_id, img_name = img_line.strip().split()
    img_id = int(img_id)

    if split_dict[img_id] == 1:
        bbox = list(map(float, bbox_line.strip().split()[1:]))

        img_name2 = os.path.join("images", img_name)
        img_path = os.path.join(root_dir, img_name2)
        xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]
        
        xmin, ymin, xmax, ymax = adjust(xmin, ymin, xmax, ymax)
        im = Image.open(img_path).convert("RGB")
        im = im.crop([xmin, ymin, xmax, ymax])

        im.save(os.path.join(root_dir, "squarecroppedimages", img_name))