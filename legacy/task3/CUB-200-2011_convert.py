from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation
from sahi.utils.file import save_json
from PIL import Image
import os

root_dir = "/mnt/sda/2022-0526/home/scc/zty/code/ANN/finalproject/new_test_project/data/CUB-200-2011/CUB_200_2011"

# Set up an empty Coco object and categories
coco = Coco()
coco.add_category(CocoCategory(id=0, name='bird'))

# Get list of image files, image size, and corresponding bounding boxes
with open(os.path.join(root_dir, "images.txt"), 'r') as f:
    img_list = f.readlines()

with open(os.path.join(root_dir, "bounding_boxes.txt"), 'r') as f:
    bbox_list = f.readlines()

with open(os.path.join(root_dir, "image_class_labels.txt"), 'r') as f:
    class_list = f.readlines()

# Create a dictionary for image split
split_dict = {}
with open(os.path.join(root_dir, "train_test_split.txt"), 'r') as f:
    split_list = f.readlines()
    for line in split_list:
        img_id, is_train = line.strip().split()
        split_dict[int(img_id)] = int(is_train)

# Create CocoImage and CocoAnnotation objects for each image
for img_line, bbox_line, class_line in zip(img_list, bbox_list, class_list):
    img_id, img_name = img_line.strip().split()
    img_id = int(img_id)

    bbox = list(map(float, bbox_line.strip().split()[1:]))

    img_name = os.path.join("images", img_name)
    img_path = os.path.join(root_dir, img_name)
    Width, Height = Image.open(img_path).size

    coco_image = CocoImage(id=img_id, file_name=img_name, height=Height, width=Width)
    coco_image.add_annotation(CocoAnnotation(
        image_id=img_id,
        bbox=bbox,
        category_id=0,
        category_name="bird"))

    coco.add_image(coco_image)

# Split data into training and testing sets
coco_train = Coco()
coco_test = Coco()

for category in coco.categories:
    coco_train.add_category(category)
    coco_test.add_category(category)

for image in coco.images:
    if split_dict[image.id] == 1:
        coco_train.add_image(image)
    else:
        coco_test.add_image(image)

# Save to COCO json format
save_json(data=coco_train.json, save_path=os.path.join(root_dir, "coco_train.json"))
save_json(data=coco_test.json, save_path=os.path.join(root_dir, "coco_test.json"))