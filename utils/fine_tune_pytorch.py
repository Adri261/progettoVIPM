import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import csv
import numpy as np
from tqdm import tqdm
from IPython.display import clear_output
from sklearn.metrics import accuracy_score, confusion_matrix

def fine_tune_network_layers(cuda, model, x_train, y_train, n_epochs, batch_size, loss_function, optimizer, k_for_accuracy=5, x_val = None, y_val =None):
    in_feat = model[-1].in_features
    model[-1] = nn.Linear(in_features=in_feat, out_features= 251, bias=True)
    print("------------------Layers to fine-tune------------------")
    print(model[:])
    print("-------------------------------------------------------")
    
    if (cuda):
        model.cuda()
        y_train = torch.tensor(y_train).type(torch.LongTensor).cuda()
        y_val =torch.tensor(y_val).type(torch.LongTensor).cuda()
    else:
        y_train = torch.tensor(y_train).type(torch.LongTensor)
        x_train = torch.from_numpy(x_train).float()
    
    if x_val == None:
        dataset_train = torch.utils.data.TensorDataset(x_train, y_train)
        train_dataset, val_dataset = torch.utils.data.random_split(dataset_train, [0.9, 0.1])
        training_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    else:
        dataset_train = torch.utils.data.TensorDataset(x_train, y_train)
        val_dataset = torch.utils.data.TensorDataset(x_val, y_val)
        training_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    model, losses, accuracies, k_accuracies = train_model(training_loader, validation_loader, n_epochs, model, loss_function, optimizer, k_for_accuracy=k_for_accuracy)
    
    return model, losses, accuracies, k_accuracies

def train_model(training_loader, validation_loader, n_epochs, model, loss_function, optimizer, k_for_accuracy=5):
    
    losses = np.empty((n_epochs,2))
    accuracies = np.empty((n_epochs,2))
    k_accuracies = np.empty((n_epochs,2))
    for epoch in range(n_epochs):
        print('EPOCH {}:'.format(epoch + 1))
        print("---------------------Training---------------------")
        # Make sure gradient tracking is on, and do a pass over the data
        model.train()
        avg_loss, epoch_accuracy, train_k_accuracy = train_one_epoch(training_loader, model, loss_function, optimizer, k_for_accuracy=k_for_accuracy)

        running_vloss = 0.0
        # Set the model to evaluation mode, disabling dropout and using population
        # statistics for batch normalization.
        model.eval()
        print("--------------------------------------------------")
        print("--------------------Validation--------------------")
        i = 1
        correct = 0
        k_correct = 0
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
                _, indexes = torch.sort(voutputs, descending=True)
                indexes = indexes[:,0:k_for_accuracy]
                j = 0
                for label in vlabels:
                    if label in indexes[j]:
                        k_correct += 1
                    j+=1
                vloss = loss_function(voutputs, vlabels)
                running_vloss += vloss.item()
                i += 1
        val_accuracy = 100*correct/total
        avg_vloss = running_vloss/len(validation_loader)
        val_k_accuracy = 100*k_correct/total
        print(f'Validation Loss: {avg_vloss}, Accuracy: {val_accuracy}%, {k_for_accuracy}-Accuracy: {val_k_accuracy}%')

        
        losses[epoch] = [avg_loss, avg_vloss]
        accuracies[epoch] = [epoch_accuracy, val_accuracy]
        k_accuracies[epoch] = [train_k_accuracy, val_k_accuracy]

    return model, losses, accuracies, k_accuracies

def train_one_epoch(training_loader, model, loss_function, optimizer, k_for_accuracy):
    k = k_for_accuracy
    n_batch = 1
    running_loss = 0.0
    correct = 0
    k_correct = 0
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

        # compute accuracy for the current batch
        total += labels.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        loss = loss_function(outputs, labels)

        #compute the k-accuracy for the current batch
        _, indexes = torch.sort(outputs, descending=True)
        indexes = indexes[:,0:k]
        j = 0
        for label in labels:
            if label in indexes[j]:
                k_correct += 1
            j+=1
        # compute the loss for the current batch
        running_loss += loss.item()
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        n_batch += 1
    avg_loss = running_loss/len(training_loader)
    train_accuracy = 100*correct/total
    train_k_accuracy = 100*k_correct/total
    print(f'Training Loss: {avg_loss}, Accuracy: {train_accuracy}%, {k}-Accuracy: {train_k_accuracy}%')
    return avg_loss, train_accuracy, train_k_accuracy

def eval_model_on_test_set(model, model_name, target_dir, x_test, y_test, cuda, k_for_accuracy =5):
    if (cuda):
        y_test = torch.tensor(y_test).type(torch.LongTensor).cuda()
    else:
        y_test = torch.tensor(y_test).type(torch.LongTensor)
        x_test = torch.from_numpy(x_test).float()
    
    test_loader = DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
    return eval_model_on_test_loader(model, model_name, target_dir, test_loader, cuda, k_for_accuracy= k_for_accuracy)

def eval_model_on_test_loader(model, model_name, target_dir, test_loader, cuda, k_for_accuracy = 5):
    k = k_for_accuracy
    k_correct = 0
    total = len(test_loader)
    model.eval()
    predictions = np.zeros(len(test_loader))
    # Disable gradient computation and reduce memory consumption.
    y_test = np.zeros(len(test_loader))
    with torch.no_grad():
        i = 0
        for test_data in tqdm(test_loader):
            test_features, test_labels = test_data
            y_test[i] = test_labels.cpu()
            outputs = model(test_features)
            _, predicted = torch.max(outputs, 1)
            predictions[i] = predicted.item()
            _, indexes = torch.sort(outputs, descending=True)
            indexes = indexes[0,0:k]
            if test_labels in indexes:
                k_correct += 1
            i+=1

    k_accuracy = 100*(k_correct/total)
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: {} %".format(accuracy *100))
    print("{}-Accuracy: {}".format(k, k_accuracy))
    cm = confusion_matrix(y_test, predictions)
    cm = np.array(cm)
    np.save("./{}/model_metrics/ConfM_{}.npy".format(target_dir, model_name), cm)
    return cm

def eval_ensamble_on_test_set(ensamble, ensamble_name, target_dir, x_test, y_test, cuda, k_for_accuracy =5):
    if (cuda):
        y_test = torch.tensor(y_test).type(torch.LongTensor).cuda()
    else:
        y_test = torch.tensor(y_test).type(torch.LongTensor)
        x_test = torch.from_numpy(x_test).float()
    
    test_loader = DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
    return eval_ensamble_on_test_loader(ensamble, ensamble_name, target_dir, test_loader, cuda, k_for_accuracy= k_for_accuracy)

def eval_ensamble_on_test_loader(ensamble, ensamble_name, target_dir, test_loader, cuda, k_for_accuracy = 5):
    k = k_for_accuracy
    k_correct = 0
    total = len(test_loader)
    predictions = np.zeros(len(test_loader)).astype("int")
    y_test = np.zeros(len(test_loader)).astype("int")
    with torch.no_grad():
        i = 0
        for test_data in tqdm(test_loader):
            test_features, test_labels = test_data
            y_test[i] = test_labels.cpu()
            whole_outputs = []
            for model in ensamble:
                model.eval()
                whole_outputs.append(model(test_features))
            maxes = np.zeros(len(whole_outputs))
            argmaxes = np.zeros(len(whole_outputs)).astype(int)
            indexes = []
            j=0
            for outputs in whole_outputs:
                maxes[j] = torch.max(outputs)
                argmaxes[j] = torch.argmax(outputs)
                # _, ind = torch.sort(outputs, descending=True)
                # indexes[j] = ind[0,0:k]
                j+=1
            
            ensamble_prediction = np.argmax(maxes)
            predictions[i] = argmaxes[ensamble_prediction]
            i += 1
        
    accuracy = accuracy_score(y_test, predictions)
    print("Accuracy: {} %".format(accuracy*100))
    cm = confusion_matrix(y_test, predictions)
    cm = np.array(cm)
    np.save("./{}/model_metrics/ensamble/ConfM_{}.npy".format(target_dir, ensamble_name), cm)
    return cm

        
    return 1