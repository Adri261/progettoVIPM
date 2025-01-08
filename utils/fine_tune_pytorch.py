import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import csv
import numpy as np
from tqdm import tqdm
from IPython.display import clear_output
from sklearn.metrics import accuracy_score, confusion_matrix

def fine_tune_network_layers(cuda, model, x_train, y_train, n_epochs, batch_size, loss_function, optimizer):

    model[-1].out_features = 251
    print("------------------Layers to fine-tune------------------")
    print(model[:])
    print("-------------------------------------------------------")
    
    if (cuda):
        model.cuda()
        y_train = torch.tensor(y_train).type(torch.LongTensor).cuda()
    else:
        y_train = torch.tensor(y_train).type(torch.LongTensor)
        x_train = torch.from_numpy(x_train).float()
    
    dataset_train = torch.utils.data.TensorDataset(x_train, y_train)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset_train, [0.9, 0.1])

    training_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    validation_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model,losses = train_model(training_loader, validation_loader, n_epochs, model, loss_function, optimizer)
    
    return model, losses

def train_model(training_loader, validation_loader, n_epochs, model, loss_function, optimizer):

    losses = np.empty((n_epochs,2))
    accuracies = np.empty((n_epochs,2))
    for epoch in range(n_epochs):
        clear_output(wait=True)
        print('EPOCH {}:'.format(epoch + 1))
        print("---------------------Training---------------------")
        # Make sure gradient tracking is on, and do a pass over the data
        model.train()
        avg_loss, epoch_accuracy = train_one_epoch(training_loader, model, loss_function, optimizer)

        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()
        print("--------------------------------------------------")
        print("--------------------Validation--------------------")
        i = 1
        correct = 0
        total = 0
        # Disable gradient computation and reduce memory consumption.
        with torch.no_grad():
            for vdata in validation_loader:
                if i%10 == 0:
                    print(f'Batch {i} di {len(validation_loader)}')     
                vinputs, vlabels = vdata
                voutputs = model(vinputs)
                _,predicted = torch.max(voutputs,1)
                total += vlabels.size(0)
                correct += (predicted == vlabels).sum().item()
                vloss = loss_function(voutputs, vlabels)
                running_vloss += vloss.item()
                i += 1
        val_accuracy = 100*correct/total
        avg_vloss = running_vloss/len(validation_loader)
        print(f'Validation Loss: {avg_vloss}, Accuracy: {val_accuracy}%')

        
        losses[epoch] = [avg_loss, avg_vloss]
        accuracies[epoch] = [epoch_accuracy, val_accuracy]
        print('LOSS: train {}; valid {}'.format(avg_loss, avg_vloss))

    return model, losses

def train_one_epoch(training_loader, model, loss_function, optimizer):
    # print(model.training)
    n_batch = 1
    running_loss = 0.0
    correct = 0
    total = 0
    for data in training_loader:
        if n_batch%10 == 0:
            print(f'Batch {n_batch} di {len(training_loader)}')
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loss = loss_function(outputs, labels)

        running_loss += loss.item()
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        n_batch += 1
    avg_loss = running_loss/len(training_loader)
    train_accuracy = 100*correct/total
    print(f'Training Loss: {avg_loss}, Accuracy: {train_accuracy}%')
    return avg_loss, train_accuracy

def eval_model_on_test_set(model, model_name, target_dir, x_test, y_test, cuda):
    if (cuda):
        y_test = torch.tensor(y_test).type(torch.LongTensor).cuda()
    else:
        y_test = torch.tensor(y_test).type(torch.LongTensor)
        x_test = torch.from_numpy(x_test).float()
    
    test_loader = DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
    return eval_model_on_test_loader(model, model_name, target_dir, test_loader, cuda)

def eval_model_on_test_loader(model, model_name, target_dir, test_loader, cuda):
    model.eval()

    predictions = np.zeros(len(test_loader))
    # Disable gradient computation and reduce memory consumption.
    y_test = np.zeros(len(test_loader))
    with torch.no_grad():
        i = 0
        for test_data in tqdm(test_loader):
            test_features, test_labels = test_data
            y_test[i] = test_labels.cpu()
            predictions[i] = np.argmax(np.array(model(test_features).cpu()))
            i+=1

    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: {}".format(accuracy))
    cm = confusion_matrix(y_test, predictions)
    cm = np.array(cm)
    np.save("./{}/model_metrics/ConfM_{}.npy".format(target_dir, model_name), cm)
    return cm