import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random

class ClipEmotionDataset(Dataset):
    def __init__(self, root_dir, split="train", max_samples=None):
        """
        root_dir/
          train/1/*.png  # Class 1
          train/2/*.png  # Class 2
          …
          train/7/*.png  # Class 7
        """
        self.root = os.path.join(root_dir, split)
        self.classes = sorted(os.listdir(self.root))
        self.paths = []
        
        if max_samples:
            # Calculate samples per class
            samples_per_class = max_samples // len(self.classes)
        
        for cls in self.classes:
            folder = os.path.join(self.root, cls)
            class_paths = []
            for fn in os.listdir(folder):
                if fn.lower().endswith((".jpg",".png","jpeg")):
                    # Convert 1-7 to 0-6 by subtracting 1 from the class label
                    class_paths.append((os.path.join(folder,fn), int(cls) - 1))
                
            if max_samples:
                # Randomly sample from each class
                if len(class_paths) > samples_per_class:
                    class_paths = random.sample(class_paths, samples_per_class)
            
            self.paths.extend(class_paths)

        # Shuffle paths to avoid all samples of same class being grouped together
        random.shuffle(self.paths)

        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
              mean=(0.48145466, 0.4578275, 0.40821073),
              std=(0.26862954, 0.26130258, 0.27577711)
            )
        ])

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        path, label = self.paths[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label
