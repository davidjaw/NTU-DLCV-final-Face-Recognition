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

## Model comparison
|  | Model size | # of parameters | Accuracy on Validation | Inference time | Trained weights |
| -------- | -------- | -------- | -------- | -------- | -------- |
| Inception-ResNet | 124MB | 26,781,288 | 75.75% | ~ | [link](https://drive.google.com/file/d/1ezy3zzPXoFId2vq6tsbvurtqCQkQEOqt/view?usp=sharing) |
| SqueezeNeXt | 15MB     | 3,729,786     | 71.28% | ~ | [link](https://drive.google.com/file/d/1RVldAcPByJBN5eS551xxEAaA49Rlzv39/view?usp=sharing) |
| SqueezeNeXt(T-S) | 15MB     | 3,729,786     | ~ | ~ | ~ |

* T-S refers to Teacher-Student training strategy

## Materials
### Baseline model - Inception ResNet
* [Github repo](https://github.com/davidsandberg/facenet)
* Related paper 1: ["FaceNet: A Unified Embedding for Face Recognition and Clustering"](https://arxiv.org/abs/1503.03832)
* Related paper 2: ["Deep Face Recognition"](http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf)

### Compressed model - SqueezeNeXt
* [Github repo](https://github.com/amirgholami/SqueezeNext)
* [paper](https://arxiv.org/abs/1803.10615)

### Teacher Student training
* [Github](https://github.com/EricHe98/Teacher-Student-Training)
  * Related paper: [Distilling the Knowledge in a Neural Network(2015)](https://arxiv.org/abs/1503.02531?context=cs)
  * 他的 readme.md 似乎就有很多說明
* ICLR workshop 2017: [TRANSFERRING KNOWLEDGE TO SMALLER NETWORK
WITH CLASS-DISTANCE LOSS](https://openreview.net/pdf?id=ByXrfaGFe)
  * No code


