from PIL import Image
import os
import xml.etree.ElementTree as ET
import scipy.io
import pickle
from pathlib import Path

def adjust(x1, x2, y1, y2):
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

root_dir = "/mnt/sda/2022-0526/home/scc/zty/code/ANN/finalproject/new_test_project/data/StanfordDogs"
subfolders = [ f.path for f in os.scandir(os.path.join(root_dir,"Images")) if f.is_dir() ]

print(subfolders)
for pat in subfolders:
    os.mkdir(os.path.join(pat[:66], "croppedImages" ,pat[74:]))


data_split_file = os.path.join(root_dir, "train_list.mat")
data_split = scipy.io.loadmat(data_split_file)
file_list = data_split["file_list"]

for idx in range(len(file_list)):
    img_name = os.path.join(root_dir, "Images", file_list[idx][0][0])

    anno_name = os.path.join(root_dir, "Annotation", file_list[idx][0][0][:-4])
    tree = ET.parse(anno_name)
    root = tree.getroot()
    obj = root.find('object')
    xmin = int(obj.find('bndbox/xmin').text)
    ymin = int(obj.find('bndbox/ymin').text)
    xmax = int(obj.find('bndbox/xmax').text)
    ymax = int(obj.find('bndbox/ymax').text)
    new_x1, new_y1, new_x2, new_y2 = adjust(xmin, xmax, ymin, ymax)
    
    im = Image.open(img_name).convert("RGB")
    im = im.crop([new_x1, new_y1, new_x2, new_y2])
    filePath = Path(os.path.join(root_dir, "croppedImages", file_list[idx][0][0]))
    filePath.touch(exist_ok=True)
    im.save(os.path.join(root_dir, "croppedImages", file_list[idx][0][0]))

Info = pickle.load(open("pred_result.pkl", "rb"))
for det in Info:
    im = Image.open(det['img_path']).convert("RGB")
    if len(det['pred_instances']['bboxes'].tolist()) == 0:
        x1, y1 = 0, 0
        x2, y2 = im.size
    else:
        x1, y1, x2, y2 = det['pred_instances']['bboxes'].tolist()[0]
    new_x1, new_y1, new_x2, new_y2 = adjust(x1, x2, y1, y2)
    im = im.crop([new_x1, new_y1, new_x2, new_y2])
    filePath = Path(os.path.join(det['img_path'][:66], "croppedImages", det['img_path'][74:]))
    filePath.touch(exist_ok=True)
    im.save(os.path.join(det['img_path'][:66], "croppedImages", det['img_path'][74:]))