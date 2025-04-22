from pathlib import Path
import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

def find_classes(directory):
    directory = Path(directory)
    
    class_names = sorted(entry.name for entry in os.scandir(directory))
    if not class_names:
        raise FileNotFoundError(f"[ERROR] No valid class names can be found in {directory}")
    
    class_to_label = {}
    for idx, name in enumerate(class_names):
        class_to_label[name] = idx
    
    return class_names, class_to_label



class custom_dataset(Dataset):
    
    def __init__(self, directory):
        directory = Path(directory)
        self.classes, self.class_to_label = find_classes(directory)
        self.path_list = list(directory.glob("*/*.png"))
    
    def load_image(self, index):
        image = Image.open(self.path_list[index])
        return image
    
    def __len__(self):
        return len(self.path_list)
    
    def __getitem__(self, index):
        
        image = self.load_image(index)
        simpleTransform = transforms.Compose([transforms.ToTensor(),
                                              transforms.Resize((28,28))])
        image = simpleTransform(image)
        image = (image*2) - 1 # CxHxW, -1 to 1
        
        class_name = self.path_list[index].parent.stem
        class_label = self.class_to_label[class_name]
        
        return image, class_label
    
        

