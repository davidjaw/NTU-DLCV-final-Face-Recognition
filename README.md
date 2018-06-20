# DLCV-final
Face Recognition

* Kaggle link: [Kaggle](https://www.kaggle.com/c/2018-spring-dlcv-final-project-2/leaderboard)

## Team Members
* R06942120 [Github](https://github.com/ljn3333)
* R06921076 [Github](https://github.com/YiJingLin)
* D06921016 [Github](https://github.com/davidjaw)

## Ongoing
- training Inception-ResNet-v1 wihout **center loss** (YiJing)

## Todos

### Implementation
- [x] Implement baseline network with tensorflow
- [x] Train baseline network with good performance
- [x] Implement Squeezenext with tensorflow
- [x] Train Squeezenext network (don't care about performance here)
- [ ] Implement Teacher-Student training strategy
- [ ] Train Squeezenext network under Teacher-Student policy **with better performance** than the typical one

### Presentation
- [ ] Don't know yet.

## Model descrioption

### Baseline model - Inception ResNet
* Network borrowed from [Github repo](https://github.com/davidsandberg/facenet)
  * Related paper 1: ["FaceNet: A Unified Embedding for Face Recognition and Clustering"](https://arxiv.org/abs/1503.03832)
  * Related paper 2: ["Deep Face Recognition"](http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf)
* Center loss is utilized for preventing overfitting
  * Paper: ["A Discriminative Feature Learning Approach for Deep Face Recognition"](http://ydwen.github.io/papers/WenECCV16.pdf)
* Training policy
  1. Train without data augmentation: `python train_teacher.py --finetune_level 0`
  2. Fine-tune with basic data augmentations: `python train_teacher.py --finetune_level 1`
      * Including rotation, horizontal flip, scale, crop, hue, contrast, brightness, gray-scale
  3. Fine-tune with seaweed augmentation: `python train_teacher.py --finetune_level 2`

### Compressed model - SqueezeNeXt
* ["SqueezeNext: Hardware-Aware Neural Network Design"](https://arxiv.org/abs/1803.10615)
  * Implemented **SqNxt-23v5** following [this repo](https://github.com/amirgholami/SqueezeNext)

### Teacher Student training
* [Github](https://github.com/EricHe98/Teacher-Student-Training)
  * Related paper: [Distilling the Knowledge in a Neural Network(2015)](https://arxiv.org/abs/1503.02531?context=cs)
  * 他的 readme.md 似乎就有很多說明
* ICLR workshop 2017: [TRANSFERRING KNOWLEDGE TO SMALLER NETWORK
WITH CLASS-DISTANCE LOSS](https://openreview.net/pdf?id=ByXrfaGFe)
  * No code

## Model comparison
|  | Model size | # of parameters | Accuracy on Validation | Inference time | Trained weights |
| -------- | -------- | -------- | -------- | -------- | -------- |
| In.-Res. | 124MB | 26,781,288 | 75.75% | ~ | [link](https://drive.google.com/file/d/1ezy3zzPXoFId2vq6tsbvurtqCQkQEOqt/view?usp=sharing) |
| In.-Res. w. seaweed | 124MB | 26,781,288 | 78.85% | ~ | [link](https://drive.google.com/file/d/1LM9ikf1-Cot3nGdizhMf2vYGRFIBiZqx/view?usp=sharing) |
| SqNxt-23v5 | 15MB     | 3,729,786     | 71.28% | ~ | [link](https://drive.google.com/file/d/1RVldAcPByJBN5eS551xxEAaA49Rlzv39/view?usp=sharing) |
| SqNxt-23v5(T-S) | 15MB     | 3,729,786     | ~ | ~ | ~ |

* T-S refers to Teacher-Student training strategy

