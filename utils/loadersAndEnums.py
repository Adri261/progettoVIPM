import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import csv
from enum import Enum
from torchvision import models

class datasets(Enum):
    TRAINING_LABELED = ["train_small.csv", "train_set"]
    TRAINING_UNLABELED = ["train_unlabeled.csv", "train_set"]
    TEST = ["val_info.csv", "val_set"]
    TEST_DEGRADED = ["val_info.csv", "val_set_degraded"]

class networks(Enum):
    ALEXNET = [227, models.alexnet(pretrained=True), "AlexNet"]
    RESNET50 = [224, models.resnet50(pretrained=True), "ResNet50"]
    GOOGLENET = [224, models.googlenet(pretrained=True), "GoogLeNet"]

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

    def __init__(self, dataset, network_input_size, cuda):
        super().__init__()
        self.images_names = []
        self.labels = []
        self.cuda = cuda
        dataset = dataset.value
        annotations_file = dataset[0]
        img_dir = dataset[1]
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
        image_bgr = cv2.resize((cv2.imread(self.images_names[index], cv2.IMREAD_COLOR).astype(np.double)/255), 
                                       (self.im_size,self.im_size), 
                                        interpolation=cv2.INTER_CUBIC).astype(np.float32)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        #moveaxis serve per avere come dimensione dell'immagine (3, righe, colonne) invece di (righe, colonne, 3)
        image = np.moveaxis(image_rgb, -1, 0)
        # eventualmente si può aggiungere l'alternativa di fare random cropping dell'immagine
        label = self.labels[index]
        if(self.cuda):
            return torch.from_numpy(image).cuda(), label
        else:
            return image, label
        

