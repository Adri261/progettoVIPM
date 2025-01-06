import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import csv
from enum import Enum
from torchvision import models
import matplotlib.pyplot as plt

class datasets(Enum):
    TRAINING_LABELED = ["train_small.csv", "train_set"]
    TRAINING_UNLABELED = ["train_unlabeled.csv", "train_set"]
    TEST = ["val_info.csv", "val_set"]
    TEST_DEGRADED = ["val_info.csv", "val_set_degraded"]


#https://pytorch.org/hub/pytorch_vision_alexnet/
# All pre-trained models expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224. The images have to be loaded in to a range of [0, 1] and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
class networks(Enum):
    ALEXNET = [227, models.alexnet(pretrained=True), "AlexNet"]
    RESNET50 = [224, models.resnet50(pretrained=True), "ResNet50"]
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

    def __init__(self, dataset, network_input_size, cuda, transform=None):
        super().__init__()
        self.images_names = []
        self.labels = []
        self.transform = transform
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
        if self.transform is None:
            image = np.moveaxis(image_rgb, -1, 0)
        else:  #apply the transformation pipe         
            image = self.transform(image=image_rgb)["image"] #anche se l'input è (256, 256, 3), restutuisce in formato torch.Size([3, 256, 256]) o comunque (3, h, w) se la pipeline fa crop/altro

        
        label = self.labels[index]
        if(self.cuda):
            return torch.from_numpy(image).cuda(), label
        else:
            return image, label
        

