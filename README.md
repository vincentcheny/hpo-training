# HPO-Training using Dragonfly(multi-obj) & NNI

## Installation

```bash
# DFHB
pip install nni
wget https://github.com/vincentcheny/hpo-training/releases/download/dfhb_v1.5/DFHB-1.5-py3-none-any.whl
nnictl package install DFHB-1.5-py3-none-any.whl
```

## Search Space

### Model Parameter

|               | ResNet50/<br />MobileNet |        Xception        |       Inception        |         VGG16          | LeNet-5                |  GoogLeNet  |
| :------------ | :----------------------: | :--------------------: | :--------------------: | :--------------------: | ---------------------- | :---------: |
| BATCH_SIZE    |          [2,16]          |        [8,120]         |         [2,32]         |        [8,128]         | [10,800]               |   [8,64]    |
| LEARNING_RATE |       [5e-6,5e-2]        |      [1e-5,5e-1]       |      [1e-6,1e-2]       |      [1e-5,5e-1]       | [1e-6,1e-2]            |             |
| NUM_EPOCH     |          [1,5]           |        [10,20]         |         [2,5]          |         [3,27]         | [10,100]               |   80[1,3]   |
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

### Hardware Parameter

|                Name                 |                     Range                      |
| :---------------------------------: | :--------------------------------------------: |
|    inter_op_parallelism_threads     |                     [2,4]                      |
|    intra_op_parallelism_threads     |                     [2,6]                      |
|         max_folded_constant         |                     [2,10]                     |
|          build_cost_model           |                     [0,8]                      |
| do_common_subexpression_elimination |                     [0,1]                      |
|        do_function_inlining         |                     [0,1]                      |
|          global_jit_level           |                     [0,2]                      |
|            infer_shapes             |                     [0,1]                      |
|         place_pruned_graph          |                     [0,1]                      |
|      enable_bfloat16_sendrecv       |                     [0,1]                      |
|          cross_device_ops*          | ["NcclAllReduce", "HierarchicalCopyAllReduce"] |
|             num_packs*              |                     [0,5]                      |
|         tf_gpu_thread_mode*         |    ["global", "gpu_private", "gpu_shared"]     |

*Exclude LeNet, Xception, MobileNet and ResNet50

## Performance

|                            Model                             |                           Dataset                            | Runtime |                            Result                            |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :-----: | :----------------------------------------------------------: |
|                            VGG16                             | [Cifar10](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/cifar10) (178M) |  11hrs  | ![](https://lh3.googleusercontent.com/-qyOCs-yEHJQ/X2L7KnlzRMI/AAAAAAAAAfk/UbkRS7sl82AvEodCqZT3gwtdUQLwDMBJACK8BGAsYHg/s0/2020-09-16.png) |
| [DenseNet](https://www.kaggle.com/ratan123/aptos-2019-keras-baseline) | [Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection) (10G) |  13hrs  | ![](https://lh3.googleusercontent.com/-FOWn1hbHB5A/X2L7A9F6JdI/AAAAAAAAAfg/x5nCuyoFYvQdzWHyFB1m-D7N92sBUCSmQCK8BGAsYHg/s0/2020-09-16.png) |
|                          Inception                           | [Human Protein](https://www.kaggle.com/mathormad/inceptionv3-baseline-lb-0-379/data) (14G) |  10hrs  | ![](https://lh3.googleusercontent.com/-9fgPN9cbHnc/X2L6odzzyEI/AAAAAAAAAfY/2D60nIbPU9wZEsWavyQD0nZVpp-uORHMACK8BGAsYHg/s0/2020-09-16.png) |
|                          GoogLeNet                           |         [ImageNet](http://www.image-net.org/) (134G)         |  66hrs  | ![](https://lh3.googleusercontent.com/-03vAi9X-VUo/X40f5e7kFeI/AAAAAAAAAiM/bniQv4iz6nkP1LU1_3qo6pxkh_O8cWppgCK8BGAsYHg/s0/2020-10-18.png) |

<details>
  <summary>CUHKPrototypeTuner</summary>


|                            Model                             |                           Dataset                            | Runtime |                            Result                            |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :-----: | :----------------------------------------------------------: |
|                            VGG16                             | [Cifar10](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/cifar10) (178M) |  11hrs  | ![](https://lh3.googleusercontent.com/-qyOCs-yEHJQ/X2L7KnlzRMI/AAAAAAAAAfk/UbkRS7sl82AvEodCqZT3gwtdUQLwDMBJACK8BGAsYHg/s0/2020-09-16.png) |
| [DenseNet](https://www.kaggle.com/ratan123/aptos-2019-keras-baseline) | [Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection) (10G) |  13hrs  | ![](https://lh3.googleusercontent.com/-FOWn1hbHB5A/X2L7A9F6JdI/AAAAAAAAAfg/x5nCuyoFYvQdzWHyFB1m-D7N92sBUCSmQCK8BGAsYHg/s0/2020-09-16.png) |
|                          Inception                           | [Human Protein](https://www.kaggle.com/mathormad/inceptionv3-baseline-lb-0-379/data) (14G) |  10hrs  | ![](https://lh3.googleusercontent.com/-9fgPN9cbHnc/X2L6odzzyEI/AAAAAAAAAfY/2D60nIbPU9wZEsWavyQD0nZVpp-uORHMACK8BGAsYHg/s0/2020-09-16.png) |
|                          GoogLeNet                           |         [ImageNet](http://www.image-net.org/) (134G)         |  50hrs  | ![](https://lh3.googleusercontent.com/-InEwCSUkhxY/X0j-MxnYbwI/AAAAAAAAAbU/p8G_7Hb073shM5TbXDT6lEzxIvCoRkL5wCK8BGAsYHg/s0/2020-08-28.png) |

</details>

<details>
  <summary>Performance without GPU parameters</summary>


|   Model   |                           Dataset                            | Runtime |                            Result                            |                   Cumulative Best accuracy                   |
| :-------: | :----------------------------------------------------------: | ------- | :----------------------------------------------------------: | :----------------------------------------------------------: |
|   VGG16   | [Cifar10](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/cifar10) (178M) | 10hrs   | ![](https://lh3.googleusercontent.com/-kAz-xqmNzeU/XxklJCqzj_I/AAAAAAAAAUQ/At5eRaCFjA0InUvvmH4dFYuecFyXPQk7QCK8BGAsYHg/s512/2020-07-22.png) | ![](https://lh3.googleusercontent.com/-xrLSbvmdQvY/XxklIEuSDwI/AAAAAAAAAUM/07Z5Nr_9S4w8AwFC1go7KXF-yKKkr6UTgCK8BGAsYHg/s512/2020-07-22.png) |
|  LeNet-5  | [Cifar10 ](https://www.cs.toronto.edu/~kriz/cifar.html)(350M) | 10hrs   | ![](https://lh3.googleusercontent.com/-gI-UZfMM_oY/XxkYGv0NXyI/AAAAAAAAATk/ZKsxIovv-v06paGVeeJMaZ2YhL_GZvXGwCK8BGAsYHg/s512/2020-07-22.png) | ![](https://lh3.googleusercontent.com/-Q-012FLVO0Y/XxkYEVi7tEI/AAAAAAAAATg/IrwKZz3txNksCozuWW8OT-QL4B6Aui-9QCK8BGAsYHg/s512/2020-07-22.png) |
| Xception  | [Humpback Whale](https://www.kaggle.com/c/humpback-whale-identification) (5.7G) | 10hrs   | ![](https://lh3.googleusercontent.com/-Sxxftb3bnfg/XxnT8tG81OI/AAAAAAAAAUg/UAKlCL6DJuINCmJ41ZIez4EE04DdDzd3gCK8BGAsYHg/s512/2020-07-23.png) | ![](https://lh3.googleusercontent.com/-AL-CRndM2x0/XxnT7x1PFFI/AAAAAAAAAUc/Ba6fZdZGV7AsY7wyjaY9qnWPDFsGNUWZQCK8BGAsYHg/s512/2020-07-23.png) |
| MobileNet | [Plant Leaves](https://www.tensorflow.org/datasets/catalog/plant_leaves) (6.8G) | 10hrs   | ![](https://lh3.googleusercontent.com/-8RKoBF04W6g/XxkknTOt4pI/AAAAAAAAAT8/Zlk_jWibDL0AcT4KvbemdX6KRw70wPNswCK8BGAsYHg/s512/2020-07-22.png) | ![](https://lh3.googleusercontent.com/-6VJY6WVWFVI/XxkkmYKb22I/AAAAAAAAAT4/IuB7ZZJBey04qk_a1wW35O7pUHmKv4PZgCK8BGAsYHg/s512/2020-07-22.png) |
| ResNet50  | [Plant Leaves](https://www.tensorflow.org/datasets/catalog/plant_leaves) (6.8G) | 10hrs   | ![](https://lh3.googleusercontent.com/-U5hhnRP9CaM/Xxkkb26bhLI/AAAAAAAAAT0/hFiQDKpjhcM66EpaZbTWydFoyP07laBNwCK8BGAsYHg/s512/2020-07-22.png) | ![](https://lh3.googleusercontent.com/-xdQZQUfEyOg/XxkkbCGyOQI/AAAAAAAAATw/FDsL1lbDS5MQaCKuaiz1YxJibn38mgHwACK8BGAsYHg/s512/2020-07-22.png) |
| Inception | [Human Protein](https://www.kaggle.com/mathormad/inceptionv3-baseline-lb-0-379/data) (14G) | 10hrs   | ![](https://lh3.googleusercontent.com/-4xdgF5j1_U4/XxcVzcPq57I/AAAAAAAAATE/6jNkc5Wtr_Aw5hclerGPNXpIlYUXo28LwCK8BGAsYHg/s512/2020-07-21.png) | ![](https://lh3.googleusercontent.com/-B5pYR_0it2k/XxcVyRe-8fI/AAAAAAAAATA/bOAHdueQOLIZsJGzqWRRKlOoXAqAZJ7bQCK8BGAsYHg/s512/2020-07-21.png) |
| GoogLeNet |         [ImageNet](http://www.image-net.org/) (134G)         | 50hrs   | ![](https://lh3.googleusercontent.com/-h_SmYOD-178/XyQOyNYeb-I/AAAAAAAAAWg/7pzlTXMLp2cyMTdcxPM7kTK44B4YaiclgCK8BGAsYHg/s512/2020-07-31.png) | ![](https://lh3.googleusercontent.com/-rq6bw1aUyZI/XyQOwy2LZAI/AAAAAAAAAWc/gptSvcFyxmog8VtCKpiffDmx_xuQne47wCK8BGAsYHg/s512/2020-07-31.png) |

</details>


<details>
  <summary>Unfixed-randomness Performance</summary>

|                           |    Dragonfly     |       TPE        |    Hyperband     |                            Result                            |                   Cumulative Best accuracy                   |
| :-----------------------: | :--------------: | :--------------: | :--------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|           VGG16           | 0.853 (17min20s) | 0.87 (65min21s)  | 0.826 (6min53s)  | ![](https://lh3.googleusercontent.com/-rBBWlBI47ZE/XvMsgNYl7FI/AAAAAAAAAPQ/qQglaGHuxK8H3yBPfsjYLQ8byfXVGvA9QCK8BGAsYHg/s512/2020-06-24.png) | ![](https://lh3.googleusercontent.com/-dnw077p5pCM/Xu8QbwcV73I/AAAAAAAAANk/8W2gsUGNMBYmYmCcBnyPoU6itFGdVjLFgCK8BGAsYHg/s512/2020-06-21.png) |
|          LeNet-5          |  0.645 (1min5s)  | 0.652 (1min21s)  |  0.613 (2min2s)  | ![](https://lh3.googleusercontent.com/-Zwp1028BOks/XvMsZkG6FVI/AAAAAAAAAPM/AgUmmyJH8zUcgdFLUlT8-br0J823nOxKwCK8BGAsYHg/s512/2020-06-24.png) | ![](https://lh3.googleusercontent.com/-Bo22LOKSOO0/XvEEBGtQpVI/AAAAAAAAAOE/FHksoSUg7WcERRFlJPShSQST0ovau7wZACK8BGAsYHg/s512/2020-06-22.png) |
|  Inception (Dog Breeds)   | 0.878 (32min42s) | 0.866 (15min15s) | 0.889 (32min8s)  | ![](https://lh3.googleusercontent.com/-dmCMjiPqu8M/XvMsQgqY5pI/AAAAAAAAAPI/4UxL-CaywQsRJb17bP1S96UcMFaRWAFxQCK8BGAsYHg/s512/2020-06-24.png) | ![](https://lh3.googleusercontent.com/-g7AWvZQ5YF8/Xuu7IxlwPdI/AAAAAAAAAhw/L34Sw9Z0jv0xrg8BRSC9RKfogI3ziXWowCK8BGAsYHg/s512/2020-06-18.png) |
| Inception (Human Protein) | 0.55 (33min23s)  | 0.514 (49min41s) | 0.504 (25min37s) | ![](https://lh3.googleusercontent.com/-RrIW_LWbZtg/XvhHkLlpKSI/AAAAAAAAAPo/9pHOJIdV8KUSwP0d5ow4C9A2_ApgRs9VgCK8BGAsYHg/s512/2020-06-28.png) | ![](https://lh3.googleusercontent.com/-RBEETTccvK0/XvhHiwwqlDI/AAAAAAAAAPk/OJTEzU_XlWk4_EDbSfnH8-HCFAgOhEbCACK8BGAsYHg/s512/2020-06-28.png) |
|         ResNet50          | 0.924 (77min21s) | 0.923 (75min21s) | 0.886 (67min5s)  | ![](https://lh3.googleusercontent.com/-9pIHqTL3Zi0/XvMr-gHilXI/AAAAAAAAAPA/iXxC7JbekYEE1uUDvAMi1p9bL0gz06DnwCK8BGAsYHg/s512/2020-06-24.png) | ![](https://lh3.googleusercontent.com/-0o4gDW65aQ8/Xuu7X9KZ1JI/AAAAAAAAAh4/Zg9fmmxLAAklY1yr509itEPjphfURw5tQCK8BGAsYHg/s512/2020-06-18.png) |

</details>