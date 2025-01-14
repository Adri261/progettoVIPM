import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import albumentations as A
def visualize_image(image_tensor):
    
    
    image_np = np.moveaxis(torch.squeeze(image_tensor).numpy(), 0,-1)

    for i in range(3): # Assuming the last dimension is the channel dimension 
        channel = image_np[..., i] 
        min_val = channel.min() 
        max_val = channel.max() 
        image_np[..., i] = (channel - min_val) / (max_val - min_val) 

    image_np = (image_np * 255).astype('uint8')
    
    plt.figure()
    plt.imshow(image_np)
    
    plt.axis('off') # Hide the axis plt.show()

def visualize_some_image_in_loader(loader, numOfImagesToshow):
    numOfImagesToshow=5
    i=0
    for image, y in tqdm(loader):
        i=i+1
        if i>numOfImagesToshow: break    
        visualize_image(image)