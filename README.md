# HPO-Training using Dragonfly(multi-obj) & NNI

## Configuration

|   Model   |                           Dataset                            |                      Tuning Parameter*                       | Runtime |
| :-------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :-----: |
|   VGG16   | [Cifar10](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/cifar10) (178M) | BATCH_SIZE=[8,16,32,64,128], <br />LEARNING_RATE=[5e-3,1e-3,5e-4,1e-4,5e-5], <br />DROP_OUT=[1e-1,2e-1,3e-1,4e-1,5e-1], <br />DENSE_UNIT=[32,64,128,256,512], <br />OPTIMIZER=["adam","grad","rmsp"], <br />KERNEL_SIZE=[2,3,4,5], <br />TRAIN_STEPS=[100,200,300,400] | 700min  |
|  LeNet-5  | [Cifar10 ](https://www.cs.toronto.edu/~kriz/cifar.html)(350M) | BATCH_SIZE=[10, 51, 92, 133, 175, 216, 257, 298, 340, 381, 422, 463, 505, 546, 587, 628, 670, 711], <br />LEARNING_RATE=1e-6~1e-2, <br />NKERN1=5~30, <br />NKERN2=31~60 |  80min  |
| Inception | [Dog Breeds](https://www.kaggle.com/careyai/inceptionv3-full-pretrained-model-instructions/data?select=train) (366M) | BATCH_SIZE=[2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32], LEARNING_RATE=1e-6~1e-2, <br />NUM_EPOCH=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,12,18,19,20], <br />DENSE_UNIT=[16,32,64,128,256,512] |  10hrs  |
| Inception | [Human Protein](https://www.kaggle.com/mathormad/inceptionv3-baseline-lb-0-379/data) (14G) | BATCH_SIZE=[2,4,6,8,10,12,14,16,18,20,22,24,26,28,30,32], <br />LEARNING_RATE=1e-6~1e-2, <br />NUM_EPOCH=[1,2,3,4,5,6,7,8], <br />DROP_OUT=[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6], <br />DENSE_UNIT=[64,128,256,512,1024] |  10hrs  |
| ResNet50  | [Plant Leaves](https://www.tensorflow.org/datasets/catalog/plant_leaves) (6.8G) | BATCH_SIZE=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16], <br />LEARNING_RATE=2e-6~1e-2, <br />NUM_EPOCH=[1,2,3,4,5,6,7,8] |  12hrs  |
| GoogLeNet |            [ImageNet](http://www.image-net.org/)             | EPSILON=[0.1,0.3,0.5,0.7,1.0],<br />BATCH_SIZE=[8,16,32,48,64]<br />NUM_EPOCH=[1,2,3]<br />INIT_LR=[1,5e-1,3e-1,1e-1,7e-2,5e-2,3e-2,1e-2]<br />FINAL_LR=[5e-4,1e-4,5e-5,1e-5,5e-6,1e-6]<br />WEIGHT_DECAY=[2e-3,7e-4,2e-4,7e-5,2e-5] |  40hrs  |

***Tuning Parameter**: include the following hardware parameters by default: 

|              Parameter              |    Range     |
| :---------------------------------: | :----------: |
|    inter_op_parallelism_threads     |   [2,3,4]    |
|    intra_op_parallelism_threads     |   [2,4,6]    |
|         max_folded_constant         | [2,4,6,8,10] |
|          build_cost_model           | [0,2,4,6,8]  |
| do_common_subexpression_elimination |    [0,1]     |
|        do_function_inlining         |    [0,1]     |
|          global_jit_level           |   [0,1,2]    |
|            infer_shapes             |    [0,1]     |
|         place_pruned_graph          |    [0,1]     |
|      enable_bfloat16_sendrecv       |    [0,1]     |

## Performance

|                           |    Dragonfly     |       TPE        |    Hyperband     |                            Result                            |                   Cumulative Best accuracy                   |
| :-----------------------: | :--------------: | :--------------: | :--------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|           VGG16           | 0.853 (17min20s) | 0.87 (65min21s)  | 0.826 (6min53s)  | ![](https://lh3.googleusercontent.com/-rBBWlBI47ZE/XvMsgNYl7FI/AAAAAAAAAPQ/qQglaGHuxK8H3yBPfsjYLQ8byfXVGvA9QCK8BGAsYHg/s512/2020-06-24.png) | ![](https://lh3.googleusercontent.com/-dnw077p5pCM/Xu8QbwcV73I/AAAAAAAAANk/8W2gsUGNMBYmYmCcBnyPoU6itFGdVjLFgCK8BGAsYHg/s512/2020-06-21.png) |
|          LeNet-5          |  0.645 (1min5s)  | 0.652 (1min21s)  |  0.613 (2min2s)  | ![](https://lh3.googleusercontent.com/-Zwp1028BOks/XvMsZkG6FVI/AAAAAAAAAPM/AgUmmyJH8zUcgdFLUlT8-br0J823nOxKwCK8BGAsYHg/s512/2020-06-24.png) | ![](https://lh3.googleusercontent.com/-Bo22LOKSOO0/XvEEBGtQpVI/AAAAAAAAAOE/FHksoSUg7WcERRFlJPShSQST0ovau7wZACK8BGAsYHg/s512/2020-06-22.png) |
|  Inception (Dog Breeds)   | 0.878 (32min42s) | 0.866 (15min15s) | 0.889 (32min8s)  | ![](https://lh3.googleusercontent.com/-dmCMjiPqu8M/XvMsQgqY5pI/AAAAAAAAAPI/4UxL-CaywQsRJb17bP1S96UcMFaRWAFxQCK8BGAsYHg/s512/2020-06-24.png) | ![](https://lh3.googleusercontent.com/-g7AWvZQ5YF8/Xuu7IxlwPdI/AAAAAAAAAhw/L34Sw9Z0jv0xrg8BRSC9RKfogI3ziXWowCK8BGAsYHg/s512/2020-06-18.png) |
| Inception (Human Protein) | 0.55 (33min23s)  | 0.514 (49min41s) | 0.504 (25min37s) | ![](https://lh3.googleusercontent.com/-RrIW_LWbZtg/XvhHkLlpKSI/AAAAAAAAAPo/9pHOJIdV8KUSwP0d5ow4C9A2_ApgRs9VgCK8BGAsYHg/s512/2020-06-28.png) | ![](https://lh3.googleusercontent.com/-RBEETTccvK0/XvhHiwwqlDI/AAAAAAAAAPk/OJTEzU_XlWk4_EDbSfnH8-HCFAgOhEbCACK8BGAsYHg/s512/2020-06-28.png) |
|         ResNet50          | 0.924 (77min21s) | 0.923 (75min21s) | 0.886 (67min5s)  | ![](https://lh3.googleusercontent.com/-9pIHqTL3Zi0/XvMr-gHilXI/AAAAAAAAAPA/iXxC7JbekYEE1uUDvAMi1p9bL0gz06DnwCK8BGAsYHg/s512/2020-06-24.png) | ![](https://lh3.googleusercontent.com/-0o4gDW65aQ8/Xuu7X9KZ1JI/AAAAAAAAAh4/Zg9fmmxLAAklY1yr509itEPjphfURw5tQCK8BGAsYHg/s512/2020-06-18.png) |
|         GoogLeNet         |                  |                  |                  |                                                              |                                                              |

