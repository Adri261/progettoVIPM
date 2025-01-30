## Transfer Learning con ultimi 2 strati:

### Machine Learning

Linear Support Vector obtained following accuracy: 0.099601593625498
Rbf Support Vector obtained following accuracy: 0.1342231075697211
KNN with 1 neighbors obtained following accuracy: 0.09063745019920319
KNN with 10 neighbors obtained following accuracy: 0.10458167330677291
KNN with 20 neighbors obtained following accuracy: 0.12151394422310757
KNN with 50 neighbors obtained following accuracy: 0.12350597609561753
Naive Bayes obtained following accuracy: 0.12848605577689243

### Finetune

Last Layer:
200 Epochs-128 b size:

Test set results:

Accuracy: 0.49800796812749004 %
5-Accuracy: 2.788844621513944 %
CM filename: ./Transfer_Learning/model_metrics/ConfM_finetunedMobileNetV3_small_minus1_200e_128bsize_80_20_split.npy
Model saved at: ./Storage/models/Neural_models/finetunedMobileNetV3_small_minus1_200e_128bsize_80_20_split.pth

Plots:

![](./models_plots/MobilenetV3_small/loss_minus1_200e_128bsize_80_20_split.png)
![](./models_plots/MobilenetV3_small/accuracy_minus1_200e_128bsize_80_20_split.png)
![](./models_plots/MobilenetV3_small/5-accuracy_minus1_200e_128bsize_80_20_split.png)

200 Epochs-64 b size:

Test set results:

Accuracy: 0.099601593625498 %
5-Accuracy: 1.9920318725099602 %
CM filename: ./Transfer_Learning/model_metrics/ConfM_finetunedMobileNetV3_small_minus1_200e_64bsize_80_20_split.npy
Model saved at: ./Storage/models/Neural_models/finetunedMobileNetV3_small_minus1_200e_64bsize_80_20_split.pth

Plots:

![](./models_plots/MobilenetV3_small/loss_minus1_200e_64bsize_80_20_split.png)
![](./models_plots/MobilenetV3_small/accuracy_minus1_200e_64bsize_80_20_split.png)
![](./models_plots/MobilenetV3_small/5-accuracy_minus1_200e_64bsize_80_20_split.png)


Last 2 Layers:

200 Epochs 128 bsize:

Test set results:

Accuracy: 2.788844621513944 %
5-Accuracy: 11.155378486055776 %
CM filename: ./Transfer_Learning/model_metrics/ConfM_finetunedMobileNetV3_small_minus2_200e_128bsize_80_20_split.npy
Model saved at: ./Storage/models/Neural_models/finetunedMobileNetV3_small_minus2_200e_128bsize_80_20_split.pth

Plots:

![](./models_plots/MobilenetV3_small/loss_minus2_200e_128bsize_80_20_split.png)
![](./models_plots/MobilenetV3_small/accuracy_minus2_200e_128bsize_80_20_split.png)
![](./models_plots/MobilenetV3_small/5-accuracy_minus2_200e_128bsize_80_20_split.png)

200 Epochs 64 bsize:

Accuracy: 6.47410358565737 %
5-Accuracy: 18.426294820717132 %
CM filename: ./Transfer_Learning/model_metrics/ConfM_finetunedMobileNetV3_small_minus2_200e_64bsize_80_20_split.npy
Model saved at: ./Storage/models/Neural_models/finetunedMobileNetV3_small_minus2_200e_64bsize_80_20_split.pth

Plots:

![](./models_plots/MobilenetV3_small/loss_minus2_200e_64bsize_80_20_split.png)
![](./models_plots/MobilenetV3_small/accuracy_minus2_200e_64bsize_80_20_split.png)
![](./models_plots/MobilenetV3_small/5-accuracy_minus2_200e_64bsize_80_20_split.png)