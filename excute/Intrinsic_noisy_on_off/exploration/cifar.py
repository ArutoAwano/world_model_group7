import math, random, numpy as np, torch, torchvision, torchvision.transforms as transforms
from PIL import Image

def load_cifar_dataset():
    transform = transforms.Compose([transforms.ToTensor()])
    return torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

def generate_random_cifar_observation(cifar_dataset, width=32, height=32):
    image, _ = cifar_dataset[random.randint(0, len(cifar_dataset) - 1)]
    pil_image = Image.fromarray((image.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    return np.array(pil_image.resize((width, height), Image.Resampling.BILINEAR))

def create_cifar_function_simple():
    print("Loading CIFAR-10 dataset...")
    cifar_dataset = load_cifar_dataset()
    print(f"✅ Loaded {len(cifar_dataset)} CIFAR-10 images")
    return lambda: generate_random_cifar_observation(cifar_dataset)