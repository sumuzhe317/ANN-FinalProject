import os
import scipy

root_dir = "/mnt/sda/2022-0526/home/scc/ytx/ANN/Main_Project/data/StanfordDogs"
data_split_file = os.path.join(root_dir, "train_list.mat")

data_split = scipy.io.loadmat(data_split_file)
file_list = data_split["file_list"]
labels = data_split["labels"].squeeze() - 1

f1 = open('squareCroppedImages_train_annotation.txt', 'w')

for idx in range(len(file_list)):
    img_name = os.path.join("squareCroppedImages", file_list[idx][0][0])
    label = labels[idx]
    f1.write(img_name+" "+str(label)+"\n")

f1.close()

data_split_file = os.path.join(root_dir, "test_list.mat")

data_split = scipy.io.loadmat(data_split_file)
file_list = data_split["file_list"]
labels = data_split["labels"].squeeze() - 1

f1 = open('squareCroppedImages_test_annotation.txt', 'w')

for idx in range(len(file_list)):
    img_name = os.path.join("squareCroppedImages", file_list[idx][0][0])
    label = labels[idx]
    f1.write(img_name+" "+str(label)+"\n")

f1.close()