# HPO-Training

## Summary for Training on Dragonfly(2-obj) & NNI(1-obj)

| Model     | Dataset                                                      | Performance                                                  | Cumulative Best Performance                                  | Hyperband Best Accuracy | TPE Best Accuracy | Dragonfly Best Accuracy | Runtime |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------------- | ----------------- | ----------------------- | ------- |
| VGG16     | [Cifar10](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/cifar10) (178M) | ![](https://lh3.googleusercontent.com/-9KgNHArMQko/Xu8QdcJF1bI/AAAAAAAAANo/jjAY36wB_psk-r5KGuzhUF0CJAEjMk7IgCK8BGAsYHg/s512/2020-06-21.png) | ![](https://lh3.googleusercontent.com/-dnw077p5pCM/Xu8QbwcV73I/AAAAAAAAANk/8W2gsUGNMBYmYmCcBnyPoU6itFGdVjLFgCK8BGAsYHg/s512/2020-06-21.png) | 0.826 (6min53s)         | 0.87 (65min21s)   | 0.853 (17min20s)        | 700min  |
| LeNet-5   | [Cifar10 ](https://www.cs.toronto.edu/~kriz/cifar.html)(350M) | ![](https://lh3.googleusercontent.com/-xdO3JhkZkko/XvEEB0tyfvI/AAAAAAAAAOI/rpwCqKfsBks9V-0tOnDHaB1yfOuqEXcpACK8BGAsYHg/s512/2020-06-22.png) | ![](https://lh3.googleusercontent.com/-Bo22LOKSOO0/XvEEBGtQpVI/AAAAAAAAAOE/FHksoSUg7WcERRFlJPShSQST0ovau7wZACK8BGAsYHg/s512/2020-06-22.png) | 0.613 (2min2s)          | 0.652 (1min21s)   | 0.645 (1min5s)          | 80min   |
| Inception | [Dog Classification](https://www.kaggle.com/careyai/inceptionv3-full-pretrained-model-instructions/data?select=train) (366M) | ![](https://lh3.googleusercontent.com/-oNUOeGrkn2c/Xuu7HYuFORI/AAAAAAAAAhs/47V_qlgTetA2u-0D-68gkvx9OR5npeTZwCK8BGAsYHg/s512/2020-06-18.png) | ![](https://lh3.googleusercontent.com/-g7AWvZQ5YF8/Xuu7IxlwPdI/AAAAAAAAAhw/L34Sw9Z0jv0xrg8BRSC9RKfogI3ziXWowCK8BGAsYHg/s512/2020-06-18.png) | 0.889 (32min8s)         | 0.866 (15min15s)  | 0.878 (32min42s)        | 10hrs   |
| ResNet50  | [plant_leaves](https://www.tensorflow.org/datasets/catalog/plant_leaves) (6.8G) | ![](https://lh3.googleusercontent.com/-rWZ3VEDWZw4/Xuu7W8zl2tI/AAAAAAAAAh0/Jux00t4_T88yTY44bfTCe7SUPKUsBwpDgCK8BGAsYHg/s512/2020-06-18.png) | ![](https://lh3.googleusercontent.com/-0o4gDW65aQ8/Xuu7X9KZ1JI/AAAAAAAAAh4/Zg9fmmxLAAklY1yr509itEPjphfURw5tQCK8BGAsYHg/s512/2020-06-18.png) | 0.886 (67min5s)         | 0.923 (75min21s)  | 0.924 (77min21s)        | 12hrs   |





