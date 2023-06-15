# Fine-grained-Classification
The main project of Artificial Neural Network course in SYSU, solved the fine-grained visual classification task. Proudly by Tengyang Zheng, Zhongyan Zheng, Jinpeng Zhao.

## Introduction
### Step 1&2

Prepared dataloader for 2 datasets, fine-tuned on pretrained ResNet-50 model.

### Step 3

Use sahi to change annotations of two datasets to COCO format, use MMdetection framework to train the object detection model. Fine-tuning on Faster R-CNN model, using pretrained ResNet-101 as model backbone. After locating the object, we cropped it from the original picture, we done this stage by two options: one is crop the object as its original shape, and another is force the cropping area to be square. The last stage is store the cropped images and use them train and test the ResNet-50 model as Step 1-2 does before.

### Step 4

(to be completed ...)


## Introduction

**IMPORTANT**: 

Remember to set this env!

```bash
cd /path/to/this/project
export PYTHONPATH=$PWD 
```
### Baseline

A unified code for all dataset: utils/dataloader.py

**Stanford Dogs dataset**

Main Code: ./legacy/baseline/dog.py

Checkpoint: ./checkpoint/baseline/dog

Result: Tensorboard under directory ./runs/baseline/dog/

Training Command:

```bash
CUDA_VISIBLE_DEVICES=5 python legacy/baseline/dog.py --use-gpu --multi-gpu --gpu-ids 0 --dataset dog train --epoch 25 --checkpoint-path checkpoint/baseline/dog --log-path runs/baseline/dog
```

Testing Command:

```bash
CUDA_VISIBLE_DEVICES=5 python legacy/baseline/dog.py --use-gpu --multi-gpu --gpu-ids 0 --dataset dog test --resume checkpoint/baseline/dog/model_best.pth.tar --log-path runs/baseline/dog
```

**CUB-200-2011 dataset**

Main Code: ./legacy/baseline/bird.py

Checkpoint: ./checkpoint/baseline/bird

Result: Tensorboard under directory ./runs/baseline/bird/

Training Command:

```bash
CUDA_VISIBLE_DEVICES=4 python legacy/baseline/bird.py --use-gpu --multi-gpu --gpu-ids 0 --dataset bird train --epoch 25 --checkpoint-path checkpoint/baseline/bird --log-path runs/baseline/bird
```

Testing Command:

```bash
CUDA_VISIBLE_DEVICES=4 python legacy/baseline/bird.py --use-gpu --multi-gpu --gpu-ids 0 --dataset bird test --resume checkpoint/baseline/bird/model_best.pth.tar --log-path runs/baseline/bird
```


### Step 1&2
> Various training tricks to improve model performance

> Transfer learning: fine-tune pretrained model

A unified code for all dataset: utils/dataloader.py

**Stanford Dogs dataset**

Main Code: ./legacy/task1-2/dog.py

Checkpoint: ./checkpoint/task1-2/dog

Result: Tensorboard under directory ./runs/task1-2/dog/

Training Command:

```bash
CUDA_VISIBLE_DEVICES=5 python legacy/task1-2/dog.py --use-gpu --multi-gpu --gpu-ids 0 --dataset dog train --epoch 25 --checkpoint-path checkpoint/task1-2/dog --log-path runs/task1-2/dog
```

Testing Command:

```bash
CUDA_VISIBLE_DEVICES=5 python legacy/task1-2/dog.py --use-gpu --multi-gpu --gpu-ids 0 --dataset dog test --resume checkpoint/task1-2/dog/model_best.pth.tar --log-path runs/task1-2/dog
```

**CUB-200-2011 dataset**

Main Code: ./legacy/task1-2/bird.py

Checkpoint: ./checkpoint/task1-2/bird

Result: Tensorboard under directory ./runs/task1-2/bird/

Training Command:

```bash
CUDA_VISIBLE_DEVICES=4 python legacy/task1-2/bird.py --use-gpu --multi-gpu --gpu-ids 0 --dataset bird train --epoch 25 --checkpoint-path checkpoint/task1-2/bird --log-path runs/task1-2/bird
```

Testing Command:

```bash
CUDA_VISIBLE_DEVICES=4 python legacy/task1-2/bird.py --use-gpu --multi-gpu --gpu-ids 0 --dataset bird test --resume checkpoint/task1-2/bird/model_best.pth.tar --log-path runs/task1-2/bird
```

### Step 3
> Attend to local regions: object localization or segmentation

**Stanford Dogs dataset**

Annotation convert code: ./legacy/task3/StanfordDogs_convert.py

MMdetection training config(please refer to document of MMDetection for detailed meaning): ./MMDetection/StanfordDogs_train.py

Object Detection Result: Tensorboard under directory ./runs/task3/dog/objectdetection, more images can be found under ./runs/task3/dog/objectdetection/vis_image

Original Dataset Pictures: ./data/StanfordDogs/Images

Crop Dataset code: ./legacy/task3/StanfordDogs_crop.py

Cropped Dataset Pictures: ./data/StanfordDogs/croppedImages

Square Crop Dataset code: ./legacy/task3/StanfordDogs_squareCrop.py

Square Cropped Dataset Pictures: ./data/StanfordDogs/squareCroppedImages

Train & Set checkpoint for cropped pictures: Tensorboard under directory ./runs/task3/dog/cropClassification

Train & Set checkpoint for square cropped pictures: Tensorboard under directory ./runs/task3/dog/squareCropClassification

Train & Set result for cropped pictures: Tensorboard under directory ./runs/task3/dog/cropClassification

Train & Set result for square cropped pictures: Tensorboard under directory ./runs/task3/dog/squareCropClassification

Cropped Training Command:

```bash
CUDA_VISIBLE_DEVICES=4 python legacy/task3/dog.py --use-gpu --multi-gpu --gpu-ids 0 --dataset dog_cropped train --epoch 16 --checkpoint-path checkpoint/task3/dog/cropClassification --log-path runs/task3/dog/cropClassification
```
Square Cropped Training Command:

```bash
CUDA_VISIBLE_DEVICES=4 python legacy/task3/dog.py --use-gpu --multi-gpu --gpu-ids 0 --dataset dog_square_cropped train --epoch 16 --checkpoint-path checkpoint/task3/dog/squareCropClassification --log-path runs/task3/dog/squareCropClassification
```

Cropped Testing Command:

```bash
CUDA_VISIBLE_DEVICES=4 python legacy/task3/dog.py --use-gpu --multi-gpu --gpu-ids 0 --dataset dog_cropped test --epoch 16 --checkpoint-path checkpoint/task3/dog/cropClassification --log-path runs/task3/dog/cropClassification --resume checkpoint/task3/dog/cropClassification/model_best.pth.tar
```
Square Cropped Testing Command:

```bash
CUDA_VISIBLE_DEVICES=4 python legacy/task3/dog.py --use-gpu --multi-gpu --gpu-ids 0 --dataset dog_square_cropped train --epoch 16 --checkpoint-path checkpoint/task3/dog/squareCropClassification --log-path runs/task3/dog/squareCropClassification --resume checkpoint/task3/dog/squareCropClassification/model_best.pth.tar
```

**CUB-200-2011 dataset**

Annotation convert code: ./legacy/task3/CUB-200-2011_convert.py

MMdetection training config(please refer to document of MMDetection for detailed meaning): ./MMDetection/CUB-200-2011_train.py

Object Detection Result: Tensorboard under directory ./runs/task3/bird/objectdetection, more images can be found under ./runs/task3/bird/objectdetection/vis_image

Original Dataset Pictures: ./data/CUB-200-2011/CUB_200_2011/images

Crop Dataset code: ./legacy/task3/CUB-200-2011_crop.py

Cropped Dataset Pictures: ./data/CUB-200-2011/CUB_200_2011/croppedimages

Square Crop Dataset code: ./legacy/task3/CUB-200-2011_squareCrop.py

Square Cropped Dataset Pictures: ./data/CUB-200-2011/CUB_200_2011/squarecroppedimages

Train & Set checkpoint for cropped pictures: Tensorboard under directory ./runs/task3/bird/cropClassification

Train & Set checkpoint for square cropped pictures: Tensorboard under directory ./runs/task3/bird/squareCropClassification

Train & Set result for cropped pictures: Tensorboard under directory ./runs/task3/bird/cropClassification

Train & Set result for square cropped pictures: Tensorboard under directory ./runs/task3/bird/squareCropClassification

Cropped Training Command:

```bash
CUDA_VISIBLE_DEVICES=4 python legacy/task3/bird.py --use-gpu --multi-gpu --gpu-ids 0 --dataset bird_cropped train --epoch 16 --checkpoint-path checkpoint/task3/bird/cropClassification --log-path runs/task3/bird/cropClassification
```
Square Cropped Training Command:

```bash
CUDA_VISIBLE_DEVICES=4 python legacy/task3/bird.py --use-gpu --multi-gpu --gpu-ids 0 --dataset bird_square_cropped train --epoch 16 --checkpoint-path checkpoint/task3/bird/squareCropClassification --log-path runs/task3/bird/squareCropClassification
```

Cropped Testing Command:

```bash
CUDA_VISIBLE_DEVICES=4 python legacy/task3/bird.py --use-gpu --multi-gpu --gpu-ids 0 --dataset bird_cropped test --epoch 16 --checkpoint-path checkpoint/task3/bird/cropClassification --log-path runs/task3/bird/cropClassification --resume checkpoint/task3/bird/cropClassification/model_best.pth.tar
```
Square Cropped Testing Command:

```bash
CUDA_VISIBLE_DEVICES=4 python legacy/task3/bird.py --use-gpu --multi-gpu --gpu-ids 0 --dataset bird_square_cropped train --epoch 16 --checkpoint-path checkpoint/task3/bird/squareCropClassification --log-path runs/task3/bird/squareCropClassification --resume checkpoint/task3/bird/squareCropClassification/model_best.pth.tar
```

### Step 4
> Synthetic image generation as part of data augmentation

**Stanford Dogs dataset**

Gen Code: ./legacy/task4/dog-gen.py

Gen Figure: ./legacy/task4/figures/dog

Gen Checkpoint: ./checkpoint/task4/dog

Genning Command:
```bash
python legacy/task4/dog_gen.py train --dataset dog
```

Main Code: ./legacy/task4/dog.py

Training Command:

```bash
CUDA_VISIBLE_DEVICES=4 python legacy/task4/dog.py --use-gpu --multi-gpu --gpu-ids 0 --dataset dog_gen train --epoch 16 --checkpoint-path checkpoint/task4/dog --log-path runs/task4/dog
```

Test Command:

```bash
CUDA_VISIBLE_DEVICES=4 python legacy/task4/dog.py --use-gpu --multi-gpu --gpu-ids 0 --dataset dog_gen test --epoch 16 --checkpoint-path checkpoint/task4/dog --log-path runs/task4/dog --resume checkpoint/task4/dog/model_best.pth.tar
```

**CUB-200-2011 dataset**

Gen Code: ./legacy/task4/bird-gen.py

Gen Checkpoint: ./checkpoint/task4/bird

Gen Figure: ./legacy/task4/figures/bird

Main Code: ./legacy/task5/bird.py

Training Command:

```bash
CUDA_VISIBLE_DEVICES=2 python legacy/task4/bird.py --use-gpu --multi-gpu --gpu-ids 0 --dataset bird_gen train --epoch 16 --checkpoint-path checkpoint/task4/bird --log-path runs/task4/bird
```

Test Command:

```bash
CUDA_VISIBLE_DEVICES=2 python legacy/task4/bird.py --use-gpu --multi-gpu --gpu-ids 0 --dataset bird_gen test --epoch 16 --checkpoint-path checkpoint/task4/bird --log-path runs/task4/bird --resume checkpoint/task4/bird/model_best.pth.tar
```

### Step 5
>ViT model backbone vs. CNN backbone: explore how to effectively use ViT

**Stanford Dogs dataset**

Main Code: ./legacy/task5/dog.py

Training Command:

```bash
CUDA_VISIBLE_DEVICES=4 python legacy/task5/dog.py --use-gpu --multi-gpu --gpu-ids 0 --dataset dog train --epoch 25 --checkpoint-path checkpoint/task5/dog --log-path runs/task5/dog
```

**CUB-200-2011 dataset**

Main Code: ./legacy/task5/bird.py

Training Command:

```bash
CUDA_VISIBLE_DEVICES=3 python legacy/task5/bird.py --use-gpu --multi-gpu --gpu-ids 0 --dataset bird train --epoch 25 --checkpoint-path checkpoint/task5/bird --log-path runs/task5/bird
```

### Step 6
> Interpretation of the model: visualization of model predictions

**headmap_model**

Main Code: ./legacy/task6/heatmap_model.py

**Stanford Dogs dataset**

Main Code: ./legacy/task6/dog.py

Generate heap map command:

```bash
CUDA_VISIBLE_DEVICES=4 python legacy/task6/dog.py gen_heatmap --use-gpu --multi-gpu --gpu-ids 0 --input-img data/StanfordDogs/Images/n02109961-Eskimo_dog/n02109961_1017.jpg --resume checkpoint/task1-2/dog/model_best.pth.tar --heatmap-path legacy/task6/dog/ --dataset dog
```

**CUB-200-2011 dataset**

Main Code: ./legacy/task6/bird.py

Generate heap map command:

```bash
CUDA_VISIBLE_DEVICES=4 python legacy/task6/bird.py gen_heatmap --use-gpu --multi-gpu --gpu-ids 0 --input-img data/CUB-200-2011/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg --resume checkpoint/task1-2/bird/model_best.pth.tar --heatmap-path legacy/task6/bird/ --dataset bird
```

### Step 7
>Robustness of the model: adversarial examples as input, (optional) improve robustness

**Stanford Dogs dataset**

Main Code: ./legacy/task7/dog.py

Training Command:

```bash
CUDA_VISIBLE_DEVICES=2 python legacy/task7/dog.py --use-gpu --multi-gpu --gpu-ids 0 --checkpoint-path checkpoint/task7/dog --log-path runs/task7/dog --dataset dog --epoch 50  train
```

Testing Command:

```bash
CUDA_VISIBLE_DEVICES=2 python legacy/task7/dog.py --use-gpu --multi-gpu --gpu-ids 0 --checkpoint-path checkpoint/task7/dog --log-path runs/task7/dog --dataset dog --epoch 50 --resume checkpoint/task7/dog/model_best.pth.tar test
```

**CUB-200-2011 dataset**

Main Code: ./legacy/task7/bird.py

Training Command:

```bash
CUDA_VISIBLE_DEVICES=3 python legacy/task7/bird.py --use-gpu --multi-gpu --gpu-ids 0 --checkpoint-path checkpoint/task7/bird --log-path runs/task7/bird --dataset bird --epoch 50  train
```

Testing Command:

```bash
CUDA_VISIBLE_DEVICES=3 python legacy/task7/bird.py --use-gpu --multi-gpu --gpu-ids 0 --checkpoint-path checkpoint/task7/bird --log-path runs/task7/bird --dataset bird --epoch 50  --resume checkpoint/task7/bird/model_best.pth.tar test
```

### Step 8
>Self-supervised learning: e.g., generate a pre-trained model, and/or used as an auxiliary task

**Stanford Dogs dataset**

Main Code: ./legacy/task8/dog.py

Training Stage One Command:

```bash
CUDA_VISIBLE_DEVICES=2,4 python legacy/task8/dog.py --dataset dog_unsupervised --use-gpu --multi-gpu --gpu-ids 0,1 --checkpoint-path checkpoint/task8/dog/stage1 --log-path runs/task8/dog/stage1 --batch-size 4 train_stage_1 --epoch 22
```
Training Stage Two Command:

```bash
CUDA_VISIBLE_DEVICES=4 python legacy/task8/dog.py --dataset dog --use-gpu --multi-gpu --gpu-ids 0 --checkpoint-path checkpoint/task8/dog/stage2 --log-path runs/task8/dog/stage2 --batch-size 32 train_stage_2 --epoch 16 --resume checkpoint/task8/dog/stage1/model_stage1_epoch20.pth.tar
```

**CUB-200-2011 dataset**

Main Code: ./legacy/task8/bird.py

Training Stage One Command:

```bash
 CUDA_VISIBLE_DEVICES=2,4 python legacy/task8/bird.py --dataset bird_unsupervised --use-gpu --multi-gpu --gpu-ids 0,1 --checkpoint-path checkpoint/task8/bird/stage1 --log-path runs/task8/bird/stage1 --batch-size 4 train_stage_1 --epoch
 22
```

Training Stage Two Command:

```bash
CUDA_VISIBLE_DEVICES=2,4 python legacy/task8/bird.py --dataset bird --use-gpu --multi-gpu --gpu-ids 0,1 --checkpoint-
path checkpoint/task8/bird/stage2 --log-path runs/task8/bird/stage2 --batch-size 4 train_stage_2 --epoch 16 --resume checkpoint/task8/bird/stage1/model_stage1_epoch20.pth.tar
```


## Install Environment

(to be completed ...)

## References
- Deep Residual Learning for Image Recognition
- ImageNet Classification with Deep Convolutional Neural Networks
- MMDetection: Open MMLab Detection Toolbox and Benchmark
- Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks
- Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection


