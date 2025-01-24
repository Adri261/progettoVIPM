from torchvision.transforms import v2
import torch
import cv2

class ToFloat32:
    def __call__(self, image):
        return image.float() / 255.0

class ToInt8:
    def __call__(self, image):
        return (image * 255.0).clamp(0, 255).byte()

class AverageBlur:
    def __init__(self, kernel_size=(9, 9)):
        self.kernel_size = kernel_size

    def __call__(self, image):
        if image.dim() == 4:  # If it's a batch of images
            return torch.stack([self._apply_blur(img) for img in image])
        elif image.dim() == 3:  # If it's a single image
            return self._apply_blur(image)
        else:
            raise ValueError(f"Unsupported image dimensions: {image.dim()}")

    def _apply_blur(self, img):
        # Convert the image (Tensor) to a NumPy array and ensure it's in HWC format
        img_np = img.permute(1, 2, 0).numpy()  # Convert CHW to HWC
        # Apply OpenCV's average blur
        blurred_np = cv2.blur(img_np, self.kernel_size)
        # Convert back to a Tensor in CHW format
        return torch.from_numpy(blurred_np).permute(2, 0, 1)  # Convert HWC to CHW

f32Transform = v2.RandomApply([
    AverageBlur(kernel_size=(9, 9)),  
    v2.GaussianNoise(0, 0.25)
])

i8Transform = v2.RandomApply([
    v2.RandomPosterize(2, 1),
    v2.JPEG([20, 50])
])


validation_augment = v2.Compose([
    f32Transform,
    ToInt8(),
    i8Transform,
    ToFloat32()
])