import math
import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

def load_cifar_dataset():
    """
    Load the CIFAR-10 dataset.
    Returns:
        cifar_dataset: The loaded CIFAR-10 dataset
    """
    # Define transform
    transform = transforms.Compose([transforms.ToTensor()])
    # Download dataset if not already downloaded
    cifar_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    return cifar_dataset

def generate_random_cifar_observation(cifar_dataset, width=32, height=32):
    """
    Generate a random observation from CIFAR-10 dataset with the specified dimensions.
    The function stretches the image to fill the observation space.
    Args:
        cifar_dataset: The CIFAR-10 dataset
        width: Width of the observation
        height: Height of the observation
    Returns:
        observation: numpy array with shape (height, width, 3)
    """
    # Select a random image
    random_index = random.randint(0, len(cifar_dataset) - 1)
    image, _ = cifar_dataset[random_index]
    
    # Convert tensor to PIL Image
    image_np = image.permute(1, 2, 0).numpy()  # Change from CxHxW to HxWxC
    pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
    
    # Stretch the image to the desired dimensions
    stretched_image = pil_image.resize((width, height), Image.Resampling.BILINEAR)
    
    # Convert to numpy array format
    observation = np.array(stretched_image)
    return observation

def create_cifar_function_simple():
    """
    Create a simple get_random_cifar function using your existing code
    
    Returns:
        Function that returns random CIFAR images (32, 32, 3) uint8
    """
    print("Loading CIFAR-10 dataset...")
    cifar_dataset = load_cifar_dataset()
    print(f"✅ Loaded {len(cifar_dataset)} CIFAR-10 images")
    
    def get_random_cifar():
        """Return a random CIFAR image as numpy array (32, 32, 3) uint8"""
        return generate_random_cifar_observation(cifar_dataset, width=32, height=32)
    
    return get_random_cifar

# Simple usage for NoisyTV wrapper
def create_cifar_function_for_noisy_tv(dataset_type='cifar10'):
    """
    Create CIFAR function for NoisyTV wrapper (simplified version)
    
    Args:
        dataset_type: 'cifar10' only (using your existing code)
    
    Returns:
        Function that returns random CIFAR images
    """
    if dataset_type.lower() != 'cifar10':
        print(f"Warning: Only cifar10 supported with simple loader, got {dataset_type}")
    
    return create_cifar_function_simple()

# Example usage
if __name__ == "__main__":
    print("Testing simple CIFAR loader...")
    
    # Create the function
    get_cifar = create_cifar_function_simple()
    
    # Test it
    for i in range(3):
        img = get_cifar()
        print(f"Image {i+1}: shape={img.shape}, dtype={img.dtype}, min={img.min()}, max={img.max()}")
    
    print("✅ Simple CIFAR loader test completed!")