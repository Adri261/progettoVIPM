import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from copy import deepcopy
from utils.loadersAndEnums import ImageDataset
import os
import csv
import numpy as np
from tqdm import tqdm

def extract_features_of_dataset(dataset, dataset_type, input_size, in_features, transfer_network, data_file, cuda=False, transform=None):    
    if(os.path.exists(data_file)):
        numpy_feat = np.load(data_file).astype("float32")
        if(cuda):            
            net_features = torch.from_numpy(numpy_feat).to(device="cuda")  
        else:
            net_features=-1
        y = []
        annotations_file = dataset.value[0]
        with open(annotations_file, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                y.append(row[1])
        y = np.array(y).astype("int")
    else:
        dataset_holder = ImageDataset(dataset=dataset, network_input_size=input_size, cuda=cuda, transform=transform)
        loader = DataLoader(dataset=dataset_holder, shuffle=False, batch_size=1)
        if(cuda):    
            net_features = torch.zeros(len(dataset_holder), in_features)
            y = np.zeros(len(dataset_holder)).astype("int")
            transfer_network.eval()
            with torch.no_grad():
                i = 0               
                print("extracting features:")
                for X_batch, y_batch in tqdm(loader):
                    net_features[i]=transfer_network(X_batch)                           
                    if (dataset_type != "unlabled"):  
                        y[i] = y_batch[0]
                    i=i+1    
                    
            #The following code copies the neural features to a numpy array stored in the cpu in order to use it in sklearn(non-neural) classifiers
            numpy_feat = np.zeros(net_features.shape)
            i = 0 
            print("copying features:")
            for i in tqdm(range(len(net_features))):
                numpy_feat[i]= net_features[i].cpu().numpy()    
            np.save(data_file, numpy_feat)

        else: #non-cuda case
            
            net_features=-1
            numpy_feat = np.zeros((len(dataset_holder), in_features))
            y = np.zeros(len(dataset_holder)).astype("int")
            transfer_network.eval()
            with torch.no_grad():
                i = 0               
                print("extracting features:")
                for X_batch, y_batch in tqdm(loader):
                    numpy_feat[i]=transfer_network(X_batch)                           
                    if (dataset_type != "unlabled"):  
                        y[i] = y_batch[0]
                    i=i+1    
            np.save(data_file, numpy_feat)
    print("---------------------------------------------------------------------------------")
    print("Done feat extraction, total n° of istances in {}: {}".format(dataset_type, len(numpy_feat)))
    print("Feature vector shape of {}: {}".format(dataset_type, numpy_feat.shape))
    print("Label vector shape of {}: {}".format(dataset_type, y.shape))
    print("---------------------------------------------------------------------------------")

    return net_features, numpy_feat, y
    

def extract_features(target_dir, train_set, test_set, network, layers_to_remove, cuda, transform):
    
    dataset_name = train_set.value[1]
    if train_set.value[0] == "train_unlabeled.csv":
        dataset_name = "train_unlabaled"

    train_data_file = "./{}/neural_features/Train_{}_minus{}_{}.npy".format(target_dir, network.value[2], layers_to_remove, dataset_name)
    test_data_file = "./{}/neural_features/Test_{}_minus{}_{}.npy".format(target_dir, network.value[2], layers_to_remove, test_set.value[1])

    errore = "There are less than {} layer in the given network's classifier".format(layers_to_remove)
    net_input_size = network.value[0]
    net = deepcopy(network.value[1])
    if (cuda):
        net.cuda()
    last_layer_to_remove_pos = len(net.classifier)
    in_features = 0
    
    for layer in net.classifier[::-1]:
        if layers_to_remove != 0:
            if isinstance(layer, nn.Linear):
                layers_to_remove -= 1
                in_features = layer.in_features
            last_layer_to_remove_pos -= 1
    
    fine_tune_layers = nn.Sequential(*[net.classifier[i] for i in range(last_layer_to_remove_pos, len(net.classifier))])
    net.classifier = nn.Sequential(*[net.classifier[i] for i in range(last_layer_to_remove_pos)])
    
    net_features_train, numpy_feat_train, y_train = extract_features_of_dataset(dataset=train_set, dataset_type="Train",
                                                                                input_size=net_input_size,
                                                                                in_features = in_features,
                                                                                transfer_network=net,data_file=train_data_file, cuda=cuda, transform=transform)
    
    net_features_test, numpy_feat_test, y_test = extract_features_of_dataset(dataset=test_set, dataset_type="Test",
                                                                                input_size=net_input_size,
                                                                                in_features = in_features,
                                                                                transfer_network=net,data_file=test_data_file, cuda=cuda, transform=transform)
    
    return net_features_train, numpy_feat_train, y_train, net_features_test, numpy_feat_test, y_test, fine_tune_layers
    
