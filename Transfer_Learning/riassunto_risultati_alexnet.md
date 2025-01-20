## Transfer Learning con ultimi 2 strati:

### Machine Learning

With PCA, first 500 principal components:

Last 2 Layers:

- Linear Support Vector obtained following accuracy: 7.569721115537849 %
- Rbf Support Vector obtained following accuracy: 11.354581673306773 %
- KNN with 1 neighbors obtained following accuracy: 5.876494023904383 %
- KNN with 10 neighbors obtained following accuracy: 6.97211155378486 %
- KNN with 20 neighbors obtained following accuracy: 5.776892430278884 %
- KNN with 50 neighbors obtained following accuracy: 7.370517928286853 %
- Naive Bayes obtained following accuracy: 7.868525896414343 %


### Finetune

Last Layer:
100 Epochs-128 b size:

Test set results:

Accuracy: 0.49800796812749004 %
5-Accuracy: 1.693227091633466 %
CM filename: ./Transfer_Learning/model_metrics/ConfM_finetunedAlexNet_minus1_100e_128bsize_80_20_split.npy
Model saved at: ./Storage/models/Neural_models/finetunedAlexNet_minus1_100e_128bsize_80_20_split.pth

Plots:

![](./models_plots/Alexnet/loss_minus1_100e_128bsize_80_20_split.png)
![](./models_plots/Alexnet/accuracy_minus1_100e_128bsize_80_20_split.png)
![](./models_plots/Alexnet/5-accuracy_minus1_100e_128bsize_80_20_split.png)

100 Epochs-64 b size:

Test set results:

Accuracy: 0.398406374501992 %
5-Accuracy: 2.49003984063745 %
CM filename: ./Transfer_Learning/model_metrics/ConfM_finetunedAlexNet_minus1_100e_64bsize_80_20_split.npy
Model saved at: ./Storage/models/Neural_models/finetunedAlexNet_minus1_100e_64bsize_80_20_split.pth

Plots:

![](./models_plots/Alexnet/loss_minus1_100e_64bsize_80_20_split.png)
![](./models_plots/Alexnet/accuracy_minus1_100e_64bsize_80_20_split.png)
![](./models_plots/Alexnet/5-accuracy_minus1_100e_64bsize_80_20_split.png)


Last 2 Layers:

100 Epochs 128 bsize:

Test set results:

Accuracy: 11.752988047808765 %
5-Accuracy: 28.18725099601594 %
CM filename: ./Transfer_Learning/model_metrics/ConfM_finetunedAlexNet_minus2_100e_128bsize_80_20_split.npy
Model saved at: ./Storage/models/Neural_models/finetunedAlexNet_minus2_100e_128bsize_80_20_split.pth

Plots:

![](./models_plots/Alexnet/loss_minus2_100e_128bsize_80_20_split.png)
![](./models_plots/Alexnet/accuracy_minus2_100e_128bsize_80_20_split.png)
![](./models_plots/Alexnet/5-accuracy_minus2_100e_128bsize_80_20_split.png)

100 Epochs 64 bsize:

Accuracy: 12.450199203187251 %
5-Accuracy: 30.378486055776893 %
CM filename: ./Transfer_Learning/model_metrics/ConfM_finetunedAlexNet_minus2_100e_64bsize_80_20_split.npy
Model saved at: ./Storage/models/Neural_models/finetunedAlexNet_minus2_100e_64bsize_80_20_split.pth
Plots:

![](./models_plots/Alexnet/loss_minus2_100e_64bsize_80_20_split.png)
![](./models_plots/Alexnet/accuracy_minus2_100e_64bsize_80_20_split.png)
![](./models_plots/Alexnet/5-accuracy_minus2_100e_64bsize_80_20_split.png)

Last 3 Layers:

100 Epochs 128 bsize:

Test set results:

Accuracy: 11.45418326693227 %
5-Accuracy: 28.087649402390436 %
CM filename: ./Transfer_Learning/model_metrics/ConfM_finetunedAlexNet_minus3_100e_128bsize_80_20_split.npy
Model saved at: ./Storage/models/Neural_models/finetunedAlexNet_minus3_100e_128bsize_80_20_split.pth

Plots:

![](./models_plots/Alexnet/loss_minus3_100e_128bsize_80_20_split.png)
![](./models_plots/Alexnet/accuracy_minus3_100e_128bsize_80_20_split.png)
![](./models_plots/Alexnet/5-accuracy_minus3_100e_128bsize_80_20_split.png)

100 Epochs 64 bsize:

Test set results:

Accuracy: 11.852589641434264 %
5-Accuracy: 28.984063745019924 %
CM filename: ./Transfer_Learning/model_metrics/ConfM_finetunedAlexNet_minus3_100e_64bsize_80_20_split.npy
Model saved at: ./Storage/models/Neural_models/finetunedAlexNet_minus3_100e_64bsize_80_20_split.pth

Plots:

![](./models_plots/Alexnet/loss_minus3_100e_64bsize_80_20_split.png)
![](./models_plots/Alexnet/accuracy_minus3_100e_64bsize_80_20_split.png)
![](./models_plots/Alexnet/5-accuracy_minus3_100e_64bsize_80_20_split.png)