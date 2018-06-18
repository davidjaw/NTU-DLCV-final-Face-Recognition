# DLCV-final
Face Recognition

## Ongoing
- training Inception-ResNet-v1 wihout **center loss** (YiJing)

## Todos

### Implementation
- [x] Implement baseline network with tensorflow
- [x] Train baseline network with good performance
- [ ] Implement Squeezenext with tensorflow
- [ ] Train Squeezenext network (don't care about performance here)
- [ ] Implement Teacher-Student training strategy
- [ ] Train Squeezenext network under Teacher-Student policy **with better performance** than the typical one

### Presentation
- [ ] Don't know yet.

## Model comparison
|  | Model size | # of parameters | Accuracy on Validation |
| -------- | -------- | -------- | -------- |
| Inception-ResNet | 124MB | 26,781,288 | 75.75% |
| SqueezeNeXt | 15MB     | 3,729,786     | ~  |

## Materials
### Baseline model
* [Github repo](https://github.com/davidsandberg/facenet)
* Related paper 1: ["FaceNet: A Unified Embedding for Face Recognition and Clustering"](https://arxiv.org/abs/1503.03832)
* Related paper 2: ["Deep Face Recognition"](http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf)

### Squeezenext
* [Github repo](https://github.com/amirgholami/SqueezeNext)
* [paper](https://arxiv.org/abs/1803.10615)

### Teacher Student training
* [Github](https://github.com/EricHe98/Teacher-Student-Training)
  * Related paper: [Distilling the Knowledge in a Neural Network(2015)](https://arxiv.org/abs/1503.02531?context=cs)
  * 他的 readme.md 似乎就有很多說明
* ICLR workshop 2017: [TRANSFERRING KNOWLEDGE TO SMALLER NETWORK
WITH CLASS-DISTANCE LOSS](https://openreview.net/pdf?id=ByXrfaGFe)
  * No code


