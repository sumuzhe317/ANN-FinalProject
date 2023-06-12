from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation
from sahi.utils.file import save_json
from PIL import Image
import os
import xml.etree.ElementTree as ET
import scipy.io

root_dir = "/mnt/sda/2022-0526/home/scc/zty/code/ANN/finalproject/new_test_project/data/StanfordDogs"

coco = Coco()
coco.add_category(CocoCategory(id=0, name='dog'))

data_split_file = os.path.join(root_dir, "train_list.mat")
data_split = scipy.io.loadmat(data_split_file)
file_list = data_split["file_list"]

for idx in range(len(file_list)):
    img_name = os.path.join(root_dir, "Images", file_list[idx][0][0])
    Width, Height = Image.open(img_name).size
    coco_image = CocoImage(file_name=file_list[idx][0][0], height=Height, width=Width)

    anno_name = os.path.join(root_dir, "Annotation", file_list[idx][0][0][:-4])
    tree = ET.parse(anno_name)
    root = tree.getroot()
    obj = root.find('object')
    xmin = int(obj.find('bndbox/xmin').text)
    ymin = int(obj.find('bndbox/ymin').text)
    xmax = int(obj.find('bndbox/xmax').text)
    ymax = int(obj.find('bndbox/ymax').text)
    coco_image.add_annotation(CocoAnnotation(
        bbox=[xmin, ymin, xmax-xmin+1, ymax-ymin+1], 
        category_id=0, category_name='dog'))
    coco.add_image(coco_image)
save_json(data=coco.json, save_path=os.path.join(root_dir,"coco_train.json"))

coco = Coco()
coco.add_category(CocoCategory(id=0, name='dog'))

data_split_file = os.path.join(root_dir, "test_list.mat")
data_split = scipy.io.loadmat(data_split_file)
file_list = data_split["file_list"]

for idx in range(len(file_list)):
    img_name = os.path.join(root_dir, "Images", file_list[idx][0][0])
    Width, Height = Image.open(img_name).size
    coco_image = CocoImage(file_name=file_list[idx][0][0], height=Height, width=Width)

    anno_name = os.path.join(root_dir, "Annotation", file_list[idx][0][0][:-4])
    tree = ET.parse(anno_name)
    root = tree.getroot()
    obj = root.find('object')
    xmin = int(obj.find('bndbox/xmin').text)
    ymin = int(obj.find('bndbox/ymin').text)
    xmax = int(obj.find('bndbox/xmax').text)
    ymax = int(obj.find('bndbox/ymax').text)
    coco_image.add_annotation(CocoAnnotation(
        bbox=[xmin, ymin, xmax-xmin+1, ymax-ymin+1], 
        category_id=0, category_name='dog'))
    coco.add_image(coco_image)
save_json(data=coco.json, save_path=os.path.join(root_dir,"coco_test.json"))