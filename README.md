# DLCV-final
Face Recognition

* Kaggle link: [Kaggle](https://www.kaggle.com/c/2018-spring-dlcv-final-project-2/leaderboard)
* Team Members: [R06942120](https://github.com/ljn3333), [R06921076](https://github.com/YiJingLin), [D06921016](https://github.com/davidjaw)

## Todos

### Implementation
- [ ] More comprehensive ablation table
- [ ] Beat the 1st team !

## Model descrioption

### Baseline model - Inception ResNet
* Network borrowed from [Github repo](https://github.com/davidsandberg/facenet)
  * Related paper 1: ["FaceNet: A Unified Embedding for Face Recognition and Clustering"](https://arxiv.org/abs/1503.03832)
  * Related paper 2: ["Deep Face Recognition"](http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf)
* Center loss is utilized for preventing overfitting
  * Paper: ["A Discriminative Feature Learning Approach for Deep Face Recognition"](http://ydwen.github.io/papers/WenECCV16.pdf)
* Hard instance weighting to prevent safe-inference
    * Paper: ["Focal Loss for Dense Object Detection"](https://arxiv.org/abs/1708.02002)
* Triplet loss is also utilized on additional embedding layer as multi-task training tricks
    * Paper: ["FaceNet: A Unified Embedding for Face Recognition and Clustering"](https://arxiv.org/abs/1503.03832)
    * Code borrowed from Github: [tensorflow-triplet-loss](https://github.com/omoindrot/tensorflow-triplet-loss)
* Training policy
  1. Train without data augmentation: `python train_teacher.py --finetune_level 0`
  2. Fine-tune with basic data augmentations: `python train_teacher.py --finetune_level 1`
      * Including rotation, horizontal flip, scale, crop, hue, contrast, brightness, gray-scale
      * Center loss is included
  3. Fine-tune with seaweed augmentation: `python train_teacher.py --finetune_level 2`
      * Triplet loss is included
      * learning rate is decayed

### Compressed model - SqueezeNeXt
* ["SqueezeNext: Hardware-Aware Neural Network Design"](https://arxiv.org/abs/1803.10615)
  * Implemented **SqNxt-23v5** following [this repo](https://github.com/amirgholami/SqueezeNext)

### Teacher Student training
* [Github](https://github.com/EricHe98/Teacher-Student-Training)
  * Related paper: [Distilling the Knowledge in a Neural Network(2015)](https://arxiv.org/abs/1503.02531?context=cs)

## Model comparison

|  | Model size | # of params | P. V. | P. T. | fps | weights |
| :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: |
| In.-Res. | 124MB | 26,781,288 | 87.62% | 84.31% | 418.96 | [link](https://drive.google.com/file/d/1Rah5wttPwvI-LN_lE_NebjUJRZZfdhAx/view?usp=sharing) |
| SqNxt-23v5 | **13.7MB**     | **3,399,352**     | 71.28% | ~ | **635.68** | [link](https://drive.google.com/file/d/1RVldAcPByJBN5eS551xxEAaA49Rlzv39/view?usp=sharing) |
| SqNxt-23v5(T-S) | **13.7MB**     | **3,399,352**     | 83.94% | 79.55% | **635.68** | [link](https://drive.google.com/file/d/1r63reqSLVl1v7yviSDptEjh3_Bp1FFwi/view?usp=sharing)* |

* T-S refers to Teacher-Student training strategy
* P. V. refers to Performance on Validation set
* the T-S weight is for fine-tuning, thus contains embedding layer
    * weights for Test is [here](https://drive.google.com/file/d/1Yb2ZX-cJizsE8RFYnua-0Kr4nhcrMKWj/view?usp=sharing)

## Ablation study
| Basic A. | Seaweed | Center L. | P. N. L. | H. I. M. | Triplet L. | P. V. | P. T. |
| :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: |
| :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_multiplication_x: | ~30% |
| :heavy_check_mark: |  :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_multiplication_x: | 67.4% |
| :heavy_check_mark: |  :heavy_check_mark: | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_multiplication_x: | 72% |
| :heavy_check_mark: |  :heavy_multiplication_x: | :heavy_check_mark: | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_multiplication_x: | 75.75% |
| :heavy_check_mark: |  :heavy_check_mark: | :heavy_check_mark: | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_multiplication_x: | 78.11% |
| :heavy_check_mark: |  :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_multiplication_x: | :heavy_multiplication_x: | 81.81% | 82.45% |
| :heavy_check_mark: |  :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:| :heavy_check_mark: | :heavy_check_mark: | 87.62% | 84.31% |

* Basic A. refers to basic augmentations
* L. refers to loss
* P. N. L. refers to pre-logit norm loss
* H. I. M. refers to hard instance mining
* P. V. refers to Performance on Validation set
* P. T. refers to Performance on Test set (scores on Kaggle)

## Download trained weights for fine-tune
### Teacher network
| F. level | Accu | link |
| :--------: | :--------: | :--------: |
| 0 | ~24% | [Link](https://drive.google.com/file/d/1U-f09BeV1YZeqPTt8Hs_DgE70Y9h_rM1/view?usp=sharing) |
| 1 | ~73% | [Link](https://drive.google.com/file/d/17ct_unH0p8LsExAi2hX4PVG3Fsczfa21/view?usp=sharing) |
| 2 | ~81% | [Link](https://drive.google.com/file/d/1N6FiuA-3xj9r0uy-zXbwFaOHR8qiqza-/view?usp=sharing) |
| 2 | 87.62% | [Link](https://drive.google.com/file/d/1pWD78yJBt3P7FeMvLwczackWkDUvsIiq/view?usp=sharing) |

* F. level: fine_tune level
