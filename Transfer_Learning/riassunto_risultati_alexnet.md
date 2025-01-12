## Transfer Learning con ultimi 2 strati:

### Machine Learning

With PCA, first 500 principal components:

- Linear Support Vector obtained following accuracy: 0.07569721115537849
- Rbf Support Vector obtained following accuracy: 0.11354581673306773
- KNN with 1 neighbors obtained following accuracy: 0.05876494023904383
- KNN with 10 neighbors obtained following accuracy: 0.0697211155378486
- KNN with 20 neighbors obtained following accuracy: 0.05776892430278884
- KNN with 50 neighbors obtained following accuracy: 0.07370517928286853
- Naive Bayes obtained following accuracy: 0.07868525896414343


### Finetune

Last Layer:
100 Epochs:

Test set results:

Accuracy: 0.69721115537848604 %
10-Accuracy: 4.681274900398407 %

Plots:

![](./models_plots/Alexnet/loss_minus1_100e_80_20_split.png)
![](./models_plots/Alexnet/accuracy_minus1_100e_80_20_split.png)
![](./models_plots/Alexnet/10_accuracy_minus1_100e_80_20_split.png)


Last 2 Layers:

100 Epochs:

Test set results:

Accuracy: 11.055776892430279 %

10-Accuracy: 38.645418326693225 %

Plots:

![](./models_plots/Alexnet/loss_minus2_100e_80_20_split.png)
![](./models_plots/Alexnet/accuracy_minus2_100e_80_20_split.png)
![](./models_plots/Alexnet/10_accuracy_minus2_100e_80_20_split.png)


Last 3 Layers:

100 Epochs:

Test set results:

Accuracy: 12.649402390438247 %

10-Accuracy: 38.047808764940235 %

Plots:

![](./models_plots/Alexnet/loss_minus3_100e_80_20_split.png)
![](./models_plots/Alexnet/accuracy_minus3_100e_80_20_split.png)
![](./models_plots/Alexnet/10_accuracy_minus3_100e_80_20_split.png)

200 epochs:

Test set results:
Accuracy: 0.1294820717131474
10-Accuracy: 40.438247011952186

Plots:

![](./models_plots/Alexnet/loss_minus3_200e_80_20_split.png)
![](./models_plots/Alexnet/accuracy_minus3_200e_80_20_split.png)
![](./models_plots/Alexnet/10_accuracy_minus3_200e_80_20_split.png)