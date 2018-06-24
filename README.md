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
- [x] Implement Teacher-Student training strategy
- [ ] Train Squeezenext network under Teacher-Student policy **with better performance** than the typical one
  * testing by David

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
| :--------: | :--------: | :--------: | :--------: | :--------: | :--------: |
| In.-Res. | 124MB | 26,781,288 | 81.81% | ~ | [link](https://drive.google.com/file/d/1Rah5wttPwvI-LN_lE_NebjUJRZZfdhAx/view?usp=sharing) |
| SqNxt-23v5 | 15MB     | 3,729,786     | 71.28% | ~ | [link](https://drive.google.com/file/d/1RVldAcPByJBN5eS551xxEAaA49Rlzv39/view?usp=sharing) |
| SqNxt-23v5(T-S) | 15MB     | 3,729,786     | ~ | ~ | ~ |

* T-S refers to Teacher-Student training strategy

## Ablation study
|  | Basic A. | Gray-scale | Seaweed | Center loss | pre-logit norm | A-softmax | Performance |
| :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: |
| In.-Res. | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_multiplication_x: | ~30% |
| In.-Res. | :heavy_check_mark: | :heavy_check_mark: | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_multiplication_x: | 67.4% |
| In.-Res. | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_multiplication_x: | 72% |
| In.-Res. | :heavy_check_mark: | :heavy_check_mark: | :heavy_multiplication_x: | :heavy_check_mark: | :heavy_multiplication_x: | :heavy_multiplication_x: | 75.75% |
| In.-Res. | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_multiplication_x: | :heavy_multiplication_x: | 78.11% |
| In.-Res. | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_multiplication_x: | 81.81% |

* Basic A. refers to basic augmentations

## Download trained weights for fine-tune
### Teacher network
| F. level | Accu | link |
| :--------: | :--------: | :--------: |
| 0 | ~24% | [Link](https://drive.google.com/file/d/1U-f09BeV1YZeqPTt8Hs_DgE70Y9h_rM1/view?usp=sharing) |
| 1 | ~73% | [Link](https://drive.google.com/file/d/17ct_unH0p8LsExAi2hX4PVG3Fsczfa21/view?usp=sharing) |
| 2 | ~81% | [Link](https://drive.google.com/file/d/1N6FiuA-3xj9r0uy-zXbwFaOHR8qiqza-/view?usp=sharing) |

* F. level: fine_tune level
