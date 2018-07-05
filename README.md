# DLCV-final: Face Recognition

* Kaggle link: [Kaggle](https://www.kaggle.com/c/2018-spring-dlcv-final-project-2/leaderboard)
* Team Members: [R06942120](https://github.com/ljn3333), [R06921076](https://github.com/YiJingLin), [D06921016](https://github.com/davidjaw)
___
## Task description
1. Beat TA’s Baseline
    * You will be given a dataset of interest (with predetermined data split)
    * You are NOT allowed to apply any external dataset or techniques like transfer learning
2. Squeeze Your Model
    * Design a model (e.g., those with fewer parameters, simpler designs, compact or simplified versions) which would achieve comparable performances but save computation or storage costs.
___
## Model descrioption
### Baseline model - Inception ResNet
* Network borrowed from [Github repo](https://github.com/davidsandberg/facenet)
  * Related paper 1: ["FaceNet: A Unified Embedding for Face Recognition and Clustering"](https://arxiv.org/abs/1503.03832)
  * Related paper 2: ["Deep Face Recognition"](http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf)
* Center loss is utilized for preventing overfitting
  * Paper: ["A Discriminative Feature Learning Approach for Deep Face Recognition"](http://ydwen.github.io/papers/WenECCV16.pdf)
* Hard instance mining to prevent overfitting those easy-sample
    * Paper: ["Focal Loss for Dense Object Detection"](https://arxiv.org/abs/1708.02002)
* Triplet loss is also utilized on additional embedding layer as multi-task training tricks
    * Paper: ["FaceNet: A Unified Embedding for Face Recognition and Clustering"](https://arxiv.org/abs/1503.03832)
    * Code borrowed from Github: [tensorflow-triplet-loss](https://github.com/omoindrot/tensorflow-triplet-loss)
* Training policy - multi-stage training
  1. Train without data augmentation: `python train_teacher.py --finetune_level 0`
  2. Fine-tune with basic data augmentations: `python train_teacher.py --finetune_level 1`
      * Including rotation, horizontal flip, scale, crop, hue, contrast, brightness, gray-scale
      * Center loss is included with weighting factor 1e-5
      * pre-logit normalize with weighting 1e-5 also included in stage 2
  3. Fine-tune with seaweed augmentation: `python train_teacher.py --finetune_level 2`
      * Triplet loss is included
      * Learning rate is decayed to 5e-5
      * Weighting of center loss, PLN loss is increased to 1e-4

![](https://github.com/davidjaw/DLCV-2018-final_project/blob/master/figures/train_teacher.png)

___
### Compressed model - SqueezeNeXt
* ["SqueezeNext: Hardware-Aware Neural Network Design"](https://arxiv.org/abs/1803.10615)
  * Implemented **SqNxt-23v5** following [this repo](https://github.com/amirgholami/SqueezeNext)
___
### Teacher Student training
* [Github](https://github.com/EricHe98/Teacher-Student-Training)
  * Related paper: [Distilling the Knowledge in a Neural Network(2015)](https://arxiv.org/abs/1503.02531?context=cs)

![](https://github.com/davidjaw/DLCV-2018-final_project/blob/master/figures/train_TS.png)
___
## Result - Model comparison
|  | Model size | # of params | P. V. | P. T. | fps | weights |
| :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: |
| In.-Res. | 124MB | 26,781,288 | 88.91% | 85.59% | 418.96 | [link](https://drive.google.com/file/d/1pzFHxReVkH6Hzz3OV9HJ_5rkP4tsyokN/view?usp=sharing) |
| 2.0 SqNxt-23v5 | **13.7MB**     | **3,399,352**     | 71.28% | ~ | **635.68** | None |
| 2.0 SqNxt-23v5(T-S) | **13.7MB**     | **3,399,352**     | 85.98% | 82.48% | **635.68** | [link](https://drive.google.com/file/d/1bU1fl8vnIcRTbpR0M9xJxGUgSzM9Oj0K/view?usp=sharing) |
| 1.0 SqNxt-23v5(T-S) | **4.5MB**     | **1,106,456**     | 78.42% | 73.6% | **805.36** | [link](https://drive.google.com/file/d/1NiAuKTxad-_5WNJK3HD8VjpNNyWoTCE_/view?usp=sharing) |

* T-S refers to Teacher-Student training strategy
* P. V. refers to Performance on Validation set
* the T-S weight is for fine-tuning, thus contains weights of embedding layer


![](https://github.com/davidjaw/DLCV-2018-final_project/blob/master/figures/latent_teacher.jpg)
![](https://github.com/davidjaw/DLCV-2018-final_project/blob/master/figures/latent_TS.jpg)
![](https://github.com/davidjaw/DLCV-2018-final_project/blob/master/figures/latent_TS-light.jpg)
___
## Ablation study
| Basic A. | Seaweed | Center L. | P. N. L. | H. I. M. | Triplet L. | P. V. | P. T. |
| :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: |
| :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_multiplication_x: | ~30% |
| :heavy_check_mark: |  :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_multiplication_x: | 67.4% |
| :heavy_check_mark: |  :heavy_check_mark: | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_multiplication_x: | 72% |
| :heavy_check_mark: |  :heavy_multiplication_x: | :heavy_check_mark: | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_multiplication_x: | 75.75% |
| :heavy_check_mark: |  :heavy_check_mark: | :heavy_check_mark: | :heavy_multiplication_x: | :heavy_multiplication_x: | :heavy_multiplication_x: | 78.11% | 79.11% |
| :heavy_check_mark: |  :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_multiplication_x: | :heavy_multiplication_x: | 81.81% | 82.45% |
| :heavy_check_mark: |  :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:| :heavy_check_mark: | :heavy_check_mark: | 88.91% | 85.59% |
* Basic A. refers to basic augmentations
* L. refers to loss
* P. N. L. refers to pre-logit norm loss
* H. I. M. refers to hard instance mining
* P. V. refers to Performance on Validation set
* P. T. refers to Performance on Test set (scores on Kaggle)
___
## Usage
* Use `train_teacher.py` to train teacher network with [args](https://github.com/davidjaw/DLCV-2018-final_project/blob/master/train_teacher.py#L12-L20)
* Use `train_student.py` to train student under normal training policy with [args](https://github.com/davidjaw/DLCV-2018-final_project/blob/master/train_student.py#L12-L21)
* Use `train_TS.py` to train student network under TS leraning policy with [args](https://github.com/davidjaw/DLCV-2018-final_project/blob/master/train_TS.py#L11-L24)
    * Trained teacher is required
