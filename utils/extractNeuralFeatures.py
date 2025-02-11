import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from copy import deepcopy
from utils.loadersAndEnums import ImageDataset
import os
import csv
import numpy as np
from tqdm import tqdm

def extract_features_of_dataset(dataset, dataset_type, input_size, in_features, transfer_network, data_file, cuda=False, normalize = True, transform=None):    
    #if datafile is of mixed dataset it always must be re-computed because it could have changed
    if(os.path.exists(data_file)):
        print("Found an existing set of features in: {}".format(data_file))
        print("Loading features from file:")
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
        print("Did not find an existing set of features in: {}".format(data_file))
        
        dataset_holder = ImageDataset(dataset=[dataset.value[0], dataset.value[1]], network_input_size=input_size, cuda=cuda, transform=transform, normalize=normalize)
        loader = DataLoader(dataset=dataset_holder, shuffle=False, batch_size=1)
        if(cuda):
            if(not(isinstance(in_features, tuple))):    
                net_features = torch.zeros(len(dataset_holder), in_features)
            else:
                net_features = torch.zeros(((len(dataset_holder),) + in_features))
            y = np.zeros(len(dataset_holder)).astype("int")
            transfer_network.eval()
            with torch.no_grad():
                i = 0               
                print("Extracting features:")
                for X_batch, y_batch in tqdm(loader):
                    net_features[i]= torch.squeeze(transfer_network(X_batch))                           
                    if (dataset_type != "unlabled"):  
                        y[i] = y_batch[0]
                    i=i+1    
                    
            #The following code copies the neural features to a numpy array stored in the cpu in order to use it in sklearn(non-neural) classifiers
            numpy_feat = np.zeros(net_features.shape)
            i = 0 
            print("Copying features to cpu:")
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
                print("Extracting features directly to cpu:")
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
    

def extract_features(train_set, test_set, network, layers_to_remove, cuda, augmented=False, transform=None, middle = False):
    target_dir = "Storage/neural_features"
    normalize = True
    if augmented:
        normalize = False
        target_dir = "Storage/augmented_neural_features"
    if not os.path.exists("./{}".format(target_dir)):
        os.makedirs(target_dir)

    dataset_name = train_set.value[0].split(".")[0]
    # if train_set.value[0] == "train_unlabeled.csv":
    #     dataset_name = "train_unlabaled"
    # if train_set.value[0] == "train_mixed.csv":
    #     dataset_name = "train_mixed"
    # if train_set.value[0] == "train_set_80%.csv":
    #     dataset_name = "train_set_80%"
    # if train_set.value[0] == "validation_set_20%.csv":
    #     dataset_name = "validation_set_20%"

    

    errore = "There are less than {} layer in the given network's classifier".format(layers_to_remove)
    net_input_size = network.value[0]
    net = deepcopy(network.value[1])
    if (cuda):
        net.cuda()
    
    in_features = 0
    if middle == False:
        if transform == None:
            train_data_file = "./{}/Train_{}_minus{}_{}.npy".format(target_dir, network.value[2], layers_to_remove, dataset_name)
            test_data_file = "./{}/Test_{}_minus{}_{}.npy".format(target_dir, network.value[2], layers_to_remove, test_set.value[1])
        else:
            train_data_file = "./{}/Train_{}_minus{}_{}_augmented.npy".format(target_dir, network.value[2], layers_to_remove, dataset_name)
            test_data_file = "./{}/Test_{}_minus{}_{}_augmented.npy".format(target_dir, network.value[2], layers_to_remove, test_set.value[1])
        if network.value[2] == "GoogLeNet" or "ResNet50" in network.value[2]:
            childrens = list(net.children())
            last_layer_to_remove_pos = len(childrens)
            for layer in childrens[::-1]:
                if layers_to_remove != 0:
                    if isinstance(layer, nn.Linear):
                        layers_to_remove -= 1
                        in_features = layer.in_features

                    last_layer_to_remove_pos -= 1
            net = nn.Sequential(*[childrens[i] for i in range(last_layer_to_remove_pos)])

            fine_tune_layers = nn.Sequential(nn.Linear(in_features=1024,out_features=512),
                                nn.ReLU(),
                                nn.Linear(in_features=512,out_features=251)
                            )
            if "ResNet50" in network.value[2]:
                fine_tune_layers = nn.Sequential(nn.Linear(in_features=2048,out_features=1024),
                                    nn.ReLU(),
                                    nn.Linear(in_features=1024,out_features=512)
                                )
                # fine_tune_layers = nn.Sequential(nn.Linear(in_features=2048,out_features=251))
        else:
            last_layer_to_remove_pos = len(net.classifier)
            for layer in net.classifier[::-1]:
                if layers_to_remove != 0:
                    if isinstance(layer, nn.Linear):
                        layers_to_remove -= 1
                        in_features = layer.in_features
                    last_layer_to_remove_pos -= 1
            fine_tune_layers = nn.Sequential(*[net.classifier[i] for i in range(last_layer_to_remove_pos, len(net.classifier))])
            net.classifier = nn.Sequential(*[net.classifier[i] for i in range(last_layer_to_remove_pos)])
    else:
        if transform == None:
            train_data_file = "./{}/Train_{}_until{}_form_start_{}.npy".format(target_dir, network.value[2], layers_to_remove, dataset_name)
            test_data_file = "./{}/Test_{}_until{}_form_start_{}.npy".format(target_dir, network.value[2], layers_to_remove, test_set.value[1])
        else:
            train_data_file = "./{}/Train_{}_until{}_form_start_{}_augmented.npy".format(target_dir, network.value[2], layers_to_remove, dataset_name)
            test_data_file = "./{}/Test_{}_until{}_form_start_{}_augmented.npy".format(target_dir, network.value[2], layers_to_remove, test_set.value[1])
        if network.value[2] != "GoogLeNet" or not("ResNet50" in network.value[2]):
            childrens = list(net.features)
            last_layer_to_remove_pos = layers_to_remove
            
            fine_tune_layers = nn.Sequential(*[childrens[i] for i in range(last_layer_to_remove_pos, len(childrens))])

            net = nn.Sequential(*[childrens[i] for i in range(0,last_layer_to_remove_pos)])
            
            x = torch.randn([3,net_input_size,net_input_size]).cuda()
            in_features = tuple(list(net(x).squeeze().shape))
        else:
            childrens = list(net.children())
            last_layer_to_remove_pos = layers_to_remove
            
            fine_tune_layers = nn.Sequential(*[childrens[i] for i in range(last_layer_to_remove_pos, len(childrens))])

            net = nn.Sequential(*[childrens[i] for i in range(0,last_layer_to_remove_pos)])
            
            x = torch.randn([3,net_input_size,net_input_size]).cuda()
            in_features = tuple(list(net(x).squeeze().shape))
    
    a = 0
    net_features_train, numpy_feat_train, y_train = extract_features_of_dataset(dataset=train_set, dataset_type="Train",
                                                                                input_size=net_input_size,
                                                                                in_features = in_features,
                                                                                transfer_network=net,data_file=train_data_file, cuda=cuda, transform=transform, normalize=normalize)
    
    if "augmented" in test_set.value[1]:
        print("Will not apply augmentation to given test set because it was already augmented")
        transform = None
    net_features_test, numpy_feat_test, y_test = extract_features_of_dataset(dataset=test_set, dataset_type="Test",
                                                                                input_size=net_input_size,
                                                                                in_features = in_features,
                                                                                transfer_network=net,data_file=test_data_file, cuda=cuda, transform=transform, normalize=normalize)
    
    return net_features_train, numpy_feat_train, y_train, net_features_test, numpy_feat_test, y_test, fine_tune_layers, net


def extract_features_from_dataloader(loader, out_features, transfer_network, dataset_type):
    n_features = len(loader.dataset)
    if(not(isinstance(out_features, tuple))):   
        net_features = torch.zeros(n_features, out_features)
    else:
        net_features = torch.zeros(((n_features,) + out_features))
    y = np.zeros(n_features).astype("int")
    transfer_network.eval()
    with torch.no_grad():
        i = 0               
        print("Extracting features:")
        for X_batch, y_batch in tqdm(loader):
            net_features[i]=transfer_network(X_batch).squeeze()                           
            if (dataset_type != "unlabled"):  
                y[i] = y_batch[0]
            i=i+1   
    return net_features, y 
    