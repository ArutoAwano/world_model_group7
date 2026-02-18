import os
import sys
import pickle
import tarfile
import urllib.request
import numpy as np
import random
from PIL import Image

CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
DATA_DIR = "./data"

def download_and_extract_cifar(data_dir):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    tar_path = os.path.join(data_dir, "cifar-10-python.tar.gz")
    cifar_dir = os.path.join(data_dir, "cifar-10-batches-py")
    
    if not os.path.exists(cifar_dir):
        print("Downloading CIFAR-10...")
        urllib.request.urlretrieve(CIFAR_URL, tar_path)
        print("Extracting CIFAR-10...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=data_dir)
        # Check if extracted correctly (some tars extract to ./cifar-10-batches-py directly)

def load_cifar_batch(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict[b'data']

def load_cifar_dataset():
    """
    Load CIFAR-10 dataset using pure python/numpy.
    Returns:
        cifar_images: numpy array (50000, 32, 32, 3) uint8
    """
    download_and_extract_cifar(DATA_DIR)
    
    cifar_dir = os.path.join(DATA_DIR, "cifar-10-batches-py")
    all_data = []
    
    # Load all 5 training batches
    for i in range(1, 6):
        batch_path = os.path.join(cifar_dir, f"data_batch_{i}")
        if not os.path.exists(batch_path):
             # Try simple path if extraction differed
             batch_path = os.path.join(DATA_DIR, f"data_batch_{i}")
        
        data = load_cifar_batch(batch_path)
        all_data.append(data)
        
    all_data = np.concatenate(all_data) # (50000, 3072)
    # Reshape: (N, 3, 32, 32) -> (N, 32, 32, 3)
    all_data = all_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    
    return all_data

def generate_random_cifar_observation(cifar_data, width=32, height=32):
    # Select random image
    idx = random.randint(0, len(cifar_data) - 1)
    img = cifar_data[idx] # (32, 32, 3)
    
    if width != 32 or height != 32:
        pil_img = Image.fromarray(img)
        pil_img = pil_img.resize((width, height), Image.BILINEAR)
        img = np.array(pil_img)
        
    return img

def create_cifar_function_simple():
    """
    Create a simple get_random_cifar function.
    Returns:
        Function that returns random CIFAR images (32, 32, 3) uint8
    """
    print("Loading CIFAR-10 dataset (Pure Python)...")
    try:
        cifar_data = load_cifar_dataset()
        print(f"✅ Loaded {len(cifar_data)} CIFAR-10 images")
        
        def get_random_cifar():
            return generate_random_cifar_observation(cifar_data, 32, 32)
        return get_random_cifar
        
    except Exception as e:
        print(f"❌ Failed to load CIFAR: {e}")
        # Fallback to random noise if loading fails to prevent crash
        def get_noise():
            return np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        return get_noise

if __name__ == "__main__":
    print("Testing CIFAR loader...")
    get_cifar = create_cifar_function_simple()
    img = get_cifar()
    print(f"Generated image shape: {img.shape}, dtype: {img.dtype}")