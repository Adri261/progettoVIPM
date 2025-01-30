import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2
import numpy as np
import csv
from enum import Enum
from torchvision import models, transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import os

class datasets(Enum):
    TRAINING_LABELED = ["train_small.csv", "train_set"]
    TRAINING_UNLABELED = ["train_unlabeled.csv", "train_set"]
    TRAINING_MIXED = ["train_mixed.csv", "train_set"]
    TRAINING_MIXED_LABELSPREAD = ["train_mixed_labelspread.csv", "train_set"]
    TRAINING_80 = ["training_set_80%.csv", "train_set"]
    VALIDATION_20 = ["validation_set_20%.csv", "train_set"]
    VALIDATION_20_DEGRADED = ["validation_set_20%.csv", "val_20_augmented"]
    TEST = ["val_info.csv", "val_set"]
    TEST_DEGRADED = ["val_info.csv", "val_set_degraded"]
    TRAINING_LABELED_80 = ["training_set_80%.csv", "train_set"]
    VALIDATION_LABELED_20 = ["validation_set_20%.csv", "train_set"]
    VALIDATION_LABELED_20_DEGRADED_V2 = ["validation_set_20%.csv", "val_20_augmented_v2"]
    VALIDATION_LABELED_20_DEGRADED = ["validation_set_20%.csv", "val_20_augmented"]
    


#https://pytorch.org/hub/pytorch_vision_alexnet/
# All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
class networks(Enum):
    ALEXNET = [227, models.alexnet(pretrained=True), "AlexNet"]
    RESNET50 = [224, models.resnet50(pretrained=True), "ResNet50"]
    FOODRESNET50 = [224, models.resnet50(pretrained=True), "FoodResNet50"]
    GOOGLENET = [224, models.googlenet(pretrained=True), "GoogLeNet"]
    MOBILENET = [224, models.mobilenet_v3_small(pretrained=True), "MobileNetV3_small"]

class ImageDataset(Dataset):
    def visualize_image(image_tensor):
        print(image_tensor)
        
        image_np = np.moveaxis(torch.squeeze(image_tensor).numpy(), 0,-1)

        for i in range(3): # Assuming the last dimension is the channel dimension 
            channel = image_np[..., i] 
            min_val = channel.min() 
            max_val = channel.max() 
            image_np[..., i] = (channel - min_val) / (max_val - min_val) 

        image_np = (image_np * 255).astype('uint8')
        print(image_np)
        plt.figure()
        plt.imshow(image_np)
        print(image_np)
        plt.axis('off') # Hide the axis plt.show()

    def __init__(self, dataset, network_input_size=None, cuda=False, transform=None, y_cuda = False, normalize = False):
        super().__init__()
        self.y_cuda = y_cuda
        self.images_names = []
        self.labels = []
        self.transform = transform
        self.cuda = cuda
        self.normalize = normalize
        self.mean = torch.tensor([0.6354, 0.5413, 0.4419])
        self.std = torch.tensor([0.2760, 0.2900, 0.3161])
        if type(dataset) is list:
            annotations_file = dataset[0]
            img_dir = dataset[1]
        else:
            annotations_file = dataset.value[0]
            img_dir = dataset.value[1]
        with open(annotations_file, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                self.images_names.append("./{}/{}".format(img_dir, row[0]))
                self.labels.append(row[1])
        self.images_names = np.array(self.images_names)
        self.labels = np.array(self.labels)
        # in base al valore passato si sceglie la rete che utilizzerà il dataset, serve per modificare le dimensioni delle immagini
        self.im_size = network_input_size
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        if self.im_size is None:
            image_bgr = (cv2.imread(self.images_names[index], cv2.IMREAD_COLOR).astype(np.double)/255).astype(np.float32) 
        else:
            image_bgr = cv2.resize((cv2.imread(self.images_names[index], cv2.IMREAD_COLOR).astype(np.double)/255), 
                                        (self.im_size,self.im_size), 
                                            interpolation=cv2.INTER_CUBIC).astype(np.float32)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        #moveaxis serve per avere come dimensione dell'immagine (3, righe, colonne) invece di (righe, colonne, 3)
        
        if self.transform is None:
            image = np.moveaxis(image_rgb, -1, 0)
            if self.normalize == True:
                nrm = transforms.Normalize(mean = self.mean, std = self.std)
                image = nrm(torch.from_numpy(image))
                image = image.numpy()            
        else:  #apply the transformation pipe
            image = np.moveaxis(image_rgb, -1, 0)         
            image = torch.from_numpy(image)
            image = self.transform(image) #anche se l'input è (256, 256, 3), restutuisce in formato torch.Size([3, 256, 256]) o comunque (3, h, w) se la pipeline fa crop/altro
            image = image.numpy()
        
        label = int(self.labels[index])
        if(self.cuda):
            if self.y_cuda:
                return torch.from_numpy(image).cuda(), torch.tensor(label, dtype=torch.int64).cuda()
            else:
                return torch.from_numpy(image).cuda(), label
        else:
            return image, label
        

def dataloader_stratified_kfold(dataset, k, network_input_size, batch_size, shuffle, cuda, transform=None, y_cuda = False):
    loader_folds = []
    dataset_file = np.loadtxt(dataset.value[0], delimiter=",", dtype="str")
    skf = StratifiedKFold(n_splits=k)
    for i, (train_index, val_index) in enumerate(skf.split(dataset_file[:,0], dataset_file[:,1])):
        print(f"Fold {i}:")
        if not os.path.exists("./stratified_kFolds"):
            os.makedirs("./stratified_kFolds")
        if not os.path.exists("./stratified_kFolds/{}_folds".format(k)):
            os.makedirs("./stratified_kFolds/{}_folds".format(k))
        

        train_split = dataset_file[train_index]
        filename = "./stratified_kFolds/{}_folds/{}_train.csv".format(k,i)
        np.savetxt(filename, train_split,  delimiter = ",", fmt='%s')
        train_dataset = ImageDataset(dataset=[filename, "train_set"], network_input_size=network_input_size, cuda=cuda,
                                    transform=transform, y_cuda=y_cuda)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

        val_split = dataset_file[val_index]
        filename = "./stratified_kFolds/{}_folds/{}_val.csv".format(k,i)
        np.savetxt(filename, val_split,  delimiter = ",", fmt='%s')
        val_dataset = ImageDataset(dataset=[filename, "train_set"], network_input_size=network_input_size, cuda=cuda,
                                    transform=transform, y_cuda=y_cuda)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)

        loader_folds.append([train_dataloader, val_dataloader])
    
    return loader_folds

def separate_data_based_on_class_group(data,y_data):
    i = 0

    uova_pesce_frutti_di_mare_classes = [34,49,56,112,118,124,165,168,176,179,214,233,25,27,9,37,97,
                                         99,152,3,36,45,50,65,67,91,117,154,189,191,195,197,200,212,216,218,248]
    
    pasta_riso_classes = [16,17,23,24,54,57,70,75,79,84,87,96,132,134,142,151,153,167,180,182,
                          183,190,234,243,12,13,30,32,38,66,80,95,123,148,177,210]
    
    carne_pollo_classes = [18,20,22,28,41,42,47,71,74,77,83,85,92,93,103,104,114,122,126,
                           128,131,140,143,145,157,158,159,163,166,171,187,206,208,213,215,217,222,
                           228,230,231,240,241,242,249]
    
    torte_classes = [5,31,44,46,48,52,63,68,69,89,115,119,146,147,161,162,172,188,203,219,223,
                     224,225,227,236,237,238,239]
    
    dolci_classes = [0,1,2,8,10,21,55,62,73,86,90,101,107,109,120,125,129,136,138,139,144,
                     149,150,155,156,160,169,173,193,194,199,220,235]
    
    panini_tacos_classes = [4,6,7,14,26,29,33,43,51,81,100,102,106,127,137,164,174,184,185,198,201,
                            202,205,207,221,229,232,244,246,247]
    
    altro_classes = [11,15,35,39,40,53,58,59,60,61,64,72,76,78,82,88,94,96,105,108,100,
                     111,113,116,121,130,133,135,141,170,175,178,181,186,192,196,204,209,211,
                     226,245,250]
    
    data_uova_pesce = [[],[]]
    data_pasta_riso = [[],[]]
    data_carne_pollo = [[],[]]
    data_torte = [[],[]]
    data_dolci = [[],[]]
    data_panini = [[],[]]
    data_altro = [[],[]]

    for y in y_data:
        if y in uova_pesce_frutti_di_mare_classes:
            data_uova_pesce[0].append(data[i])
            data_uova_pesce[1].append(y)
        if y in pasta_riso_classes:
            data_pasta_riso[0].append(data[i])
            data_pasta_riso[1].append(y)
        if y in carne_pollo_classes:
            data_carne_pollo[0].append(data[i])
            data_carne_pollo[1].append(y)
        if y in torte_classes:
            data_torte[0].append(data[i])
            data_torte[1].append(y)
        if y in dolci_classes:
            data_dolci[0].append(data[i])
            data_dolci[1].append(y)
        if y in panini_tacos_classes:
            data_panini[0].append(data[i])
            data_panini[1].append(y)
        if y in altro_classes:
            data_altro[0].append(data[i])
            data_altro[1].append(y)
        
        i += 1
    
    data_uova_pesce[0] = np.array(data_uova_pesce[0])
    data_uova_pesce[1] = np.array(data_uova_pesce[1])

    data_pasta_riso[0] = np.array(data_pasta_riso[0])
    data_pasta_riso[1] = np.array(data_pasta_riso[1])

    data_carne_pollo[0] = np.array(data_carne_pollo[0])
    data_carne_pollo[1] = np.array(data_carne_pollo[1])

    data_torte[0] = np.array(data_torte[0])
    data_torte[1] = np.array(data_torte[1])

    data_dolci[0] = np.array(data_dolci[0])
    data_dolci[1] = np.array(data_dolci[1])

    data_panini[0] = np.array(data_panini[0])
    data_panini[1] = np.array(data_panini[1])

    data_altro[0] = np.array(data_altro[0])
    data_altro[1] = np.array(data_altro[1])
        
    return data_uova_pesce,data_pasta_riso, data_carne_pollo, data_torte, data_dolci, data_panini,data_altro
