# HPO-Training using Dragonfly(multi-obj) & NNI

## Basic Info

|   Model   |                           Dataset                            | Runtime |
| :-------: | :----------------------------------------------------------: | :-----: |
| ResNet50  | [Plant Leaves](https://www.tensorflow.org/datasets/catalog/plant_leaves) (6.8G) |  10hrs  |
| MobileNet | [Plant Leaves](https://www.tensorflow.org/datasets/catalog/plant_leaves) (6.8G) |  10hrs  |
| Xception  | [Humpback Whale](https://www.kaggle.com/c/humpback-whale-identification) (5.7G) |  10hrs  |
| Inception | [Human Protein](https://www.kaggle.com/mathormad/inceptionv3-baseline-lb-0-379/data) (14G) |  10hrs  |
|   VGG16   | [Cifar10](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/cifar10) (178M) |  10hrs  |
|  LeNet-5  | [Cifar10 ](https://www.cs.toronto.edu/~kriz/cifar.html)(350M) |  10hrs  |
| GoogLeNet |         [ImageNet](http://www.image-net.org/) (134G)         |  40hrs  |

## Search Space

### Model Parameter

|               | ResNet50/<br />MobileNet |        Xception        |       Inception        |         VGG16          | LeNet-5                |  GoogLeNet  |
| :------------ | :----------------------: | :--------------------: | :--------------------: | :--------------------: | ---------------------- | :---------: |
| BATCH_SIZE    |          [2,16]          |        [8,120]         |         [2,32]         |        [8,128]         | [10,800]               |   [8,64]    |
| LEARNING_RATE |       [5e-6,5e-2]        |      [1e-5,5e-1]       |      [1e-6,1e-2]       |      [1e-5,5e-1]       | [1e-6,1e-2]            |             |
| NUM_EPOCH     |          [1,5]           |        [10,20]         |         [2,5]          |         [2,30]         | [10,100]               |   80[1,3]   |
| DROP_OUT      |                          |                        |                        |                        |                        |             |
| DENSE_UNIT    |                          |                        |        [64,512]        |       [64,1024]        | [16,1024]              |             |
| OPTIMIZER     |  ["adam","grad","rmsp"]  | ["adam","grad","rmsp"] | ["adam","grad","rmsp"] | ["adam","grad","rmsp"] | ["adam","grad","rmsp"] |             |
| KERNEL_SIZE   |                          |                        |                        |         [1,5]          |                        |             |
| NKERN1        |                          |                        |                        |                        | [5,30]                 |             |
| NKERN2        |                          |                        |                        |                        | [31,60]                |             |
| EPSILON       |                          |                        |                        |                        |                        |  [0.1,1.0]  |
| INIT_LR       |                          |                        |                        |                        |                        |  [1e-2,1]   |
| FINAL_LR      |                          |                        |                        |                        |                        | [1e-6,5e-4] |
| WEIGHT_DECAY  |                          |      [1e-5,5e-2]       |                        |      [1e-5,8e-2]       |                        | [2e-5,2e-3] |
| NUM_FILTER    |                          |                        |        [16,128]        |         [8,64]         |                        |             |

### Hardware Parameter*

|                Name                 | Range  |
| :---------------------------------: | :----: |
|    inter_op_parallelism_threads     | [2,4]  |
|    intra_op_parallelism_threads     | [2,6]  |
|         max_folded_constant         | [2,10] |
|          build_cost_model           | [0,8]  |
| do_common_subexpression_elimination | [0,1]  |
|        do_function_inlining         | [0,1]  |
|          global_jit_level           | [0,2]  |
|            infer_shapes             | [0,1]  |
|         place_pruned_graph          | [0,1]  |
|      enable_bfloat16_sendrecv       | [0,1]  |

*Hardware Parameter*: Every model includes hardware parameters listed by default when run with dragonfly.

## Performance

|           |    Dragonfly     |        TPE        |       BOHB        |                            Result                            |                   Cumulative Best accuracy                   |
| :-------: | :--------------: | :---------------: | :---------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| ResNet50  | 0.743 (63.36min) | 0.942 (116.21min) | 0.849 (96.83min)  | ![](https://lh3.googleusercontent.com/-U5hhnRP9CaM/Xxkkb26bhLI/AAAAAAAAAT0/hFiQDKpjhcM66EpaZbTWydFoyP07laBNwCK8BGAsYHg/s512/2020-07-22.png) | ![](https://lh3.googleusercontent.com/-xdQZQUfEyOg/XxkkbCGyOQI/AAAAAAAAATw/FDsL1lbDS5MQaCKuaiz1YxJibn38mgHwACK8BGAsYHg/s512/2020-07-22.png) |
| MobileNet | 0.851 (93.13min) | 0.945 (119.23min) | 0.973 (181.94min) | ![](https://lh3.googleusercontent.com/-8RKoBF04W6g/XxkknTOt4pI/AAAAAAAAAT8/Zlk_jWibDL0AcT4KvbemdX6KRw70wPNswCK8BGAsYHg/s512/2020-07-22.png) | ![](https://lh3.googleusercontent.com/-6VJY6WVWFVI/XxkkmYKb22I/AAAAAAAAAT4/IuB7ZZJBey04qk_a1wW35O7pUHmKv4PZgCK8BGAsYHg/s512/2020-07-22.png) |
| Xception  | 0.685 (25.49min) | 0.995 (22.22min)  | 0.403 (33.61min)  | ![](https://lh3.googleusercontent.com/-Sxxftb3bnfg/XxnT8tG81OI/AAAAAAAAAUg/UAKlCL6DJuINCmJ41ZIez4EE04DdDzd3gCK8BGAsYHg/s512/2020-07-23.png) | ![](https://lh3.googleusercontent.com/-AL-CRndM2x0/XxnT7x1PFFI/AAAAAAAAAUc/Ba6fZdZGV7AsY7wyjaY9qnWPDFsGNUWZQCK8BGAsYHg/s512/2020-07-23.png) |
| Inception | 0.550 (33.38min) | 0.440 (105.63min) | 0.428 (30.50min)  | ![](https://lh3.googleusercontent.com/-4xdgF5j1_U4/XxcVzcPq57I/AAAAAAAAATE/6jNkc5Wtr_Aw5hclerGPNXpIlYUXo28LwCK8BGAsYHg/s512/2020-07-21.png) | ![](https://lh3.googleusercontent.com/-B5pYR_0it2k/XxcVyRe-8fI/AAAAAAAAATA/bOAHdueQOLIZsJGzqWRRKlOoXAqAZJ7bQCK8BGAsYHg/s512/2020-07-21.png) |
|   VGG16   | 0.795 (32.91min) |  0.775 (7.04min)  | 0.809 (15.34min)  | ![](https://lh3.googleusercontent.com/-kAz-xqmNzeU/XxklJCqzj_I/AAAAAAAAAUQ/At5eRaCFjA0InUvvmH4dFYuecFyXPQk7QCK8BGAsYHg/s512/2020-07-22.png) | ![](https://lh3.googleusercontent.com/-xrLSbvmdQvY/XxklIEuSDwI/AAAAAAAAAUM/07Z5Nr_9S4w8AwFC1go7KXF-yKKkr6UTgCK8BGAsYHg/s512/2020-07-22.png) |
|  LeNet-5  | 0.424 (5.15min)  |  0.474 (5.68min)  |  0.346 (0.91min)  | ![](https://lh3.googleusercontent.com/-gI-UZfMM_oY/XxkYGv0NXyI/AAAAAAAAATk/ZKsxIovv-v06paGVeeJMaZ2YhL_GZvXGwCK8BGAsYHg/s512/2020-07-22.png) | ![](https://lh3.googleusercontent.com/-Q-012FLVO0Y/XxkYEVi7tEI/AAAAAAAAATg/IrwKZz3txNksCozuWW8OT-QL4B6Aui-9QCK8BGAsYHg/s512/2020-07-22.png) |
| GoogLeNet | 0.490 (65.37min) | 0.497 (72.82min)  | 0.497 (93.32min)  | ![](https://lh3.googleusercontent.com/-nGF0qKF9r6k/XxFMMpWeY7I/AAAAAAAAAS0/GMAbuMyO5Oo-bCAI_dv206vFz9ieSXtqACK8BGAsYHg/s512/2020-07-16.png) | ![](https://lh3.googleusercontent.com/-3LQZ1tU6rKs/XxFMLEC8inI/AAAAAAAAASw/iBjQQBjmnuwLFxBmBKl6nmaigAcJKxD5wCK8BGAsYHg/s512/2020-07-16.png) |

## Unfixed-randomness Performance

|                           |    Dragonfly     |       TPE        |    Hyperband     |                            Result                            |                   Cumulative Best accuracy                   |
| :-----------------------: | :--------------: | :--------------: | :--------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|           VGG16           | 0.853 (17min20s) | 0.87 (65min21s)  | 0.826 (6min53s)  | ![](https://lh3.googleusercontent.com/-rBBWlBI47ZE/XvMsgNYl7FI/AAAAAAAAAPQ/qQglaGHuxK8H3yBPfsjYLQ8byfXVGvA9QCK8BGAsYHg/s512/2020-06-24.png) | ![](https://lh3.googleusercontent.com/-dnw077p5pCM/Xu8QbwcV73I/AAAAAAAAANk/8W2gsUGNMBYmYmCcBnyPoU6itFGdVjLFgCK8BGAsYHg/s512/2020-06-21.png) |
|          LeNet-5          |  0.645 (1min5s)  | 0.652 (1min21s)  |  0.613 (2min2s)  | ![](https://lh3.googleusercontent.com/-Zwp1028BOks/XvMsZkG6FVI/AAAAAAAAAPM/AgUmmyJH8zUcgdFLUlT8-br0J823nOxKwCK8BGAsYHg/s512/2020-06-24.png) | ![](https://lh3.googleusercontent.com/-Bo22LOKSOO0/XvEEBGtQpVI/AAAAAAAAAOE/FHksoSUg7WcERRFlJPShSQST0ovau7wZACK8BGAsYHg/s512/2020-06-22.png) |
|  Inception (Dog Breeds)   | 0.878 (32min42s) | 0.866 (15min15s) | 0.889 (32min8s)  | ![](https://lh3.googleusercontent.com/-dmCMjiPqu8M/XvMsQgqY5pI/AAAAAAAAAPI/4UxL-CaywQsRJb17bP1S96UcMFaRWAFxQCK8BGAsYHg/s512/2020-06-24.png) | ![](https://lh3.googleusercontent.com/-g7AWvZQ5YF8/Xuu7IxlwPdI/AAAAAAAAAhw/L34Sw9Z0jv0xrg8BRSC9RKfogI3ziXWowCK8BGAsYHg/s512/2020-06-18.png) |
| Inception (Human Protein) | 0.55 (33min23s)  | 0.514 (49min41s) | 0.504 (25min37s) | ![](https://lh3.googleusercontent.com/-RrIW_LWbZtg/XvhHkLlpKSI/AAAAAAAAAPo/9pHOJIdV8KUSwP0d5ow4C9A2_ApgRs9VgCK8BGAsYHg/s512/2020-06-28.png) | ![](https://lh3.googleusercontent.com/-RBEETTccvK0/XvhHiwwqlDI/AAAAAAAAAPk/OJTEzU_XlWk4_EDbSfnH8-HCFAgOhEbCACK8BGAsYHg/s512/2020-06-28.png) |
|         ResNet50          | 0.924 (77min21s) | 0.923 (75min21s) | 0.886 (67min5s)  | ![](https://lh3.googleusercontent.com/-9pIHqTL3Zi0/XvMr-gHilXI/AAAAAAAAAPA/iXxC7JbekYEE1uUDvAMi1p9bL0gz06DnwCK8BGAsYHg/s512/2020-06-24.png) | ![](https://lh3.googleusercontent.com/-0o4gDW65aQ8/Xuu7X9KZ1JI/AAAAAAAAAh4/Zg9fmmxLAAklY1yr509itEPjphfURw5tQCK8BGAsYHg/s512/2020-06-18.png) |

