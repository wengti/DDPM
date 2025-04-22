from pathlib import Path
import _csv as csv
import numpy as np
import cv2

def extract_mnist(csv_fname, save_dir):
    
    save_dir = Path(save_dir)
    if not save_dir.is_dir():
        save_dir.mkdir(parents = True,
                       exist_ok = True)
    
    with open(csv_fname, "r") as f:
        reader = csv.reader(f)
        
        for idx, row in enumerate(reader):
            
            if idx == 0:
                continue
            
            img = np.zeros((28*28))
            img[:] = list(map(int, row[1:]))
            img = img.reshape((28,28))
            
            save_class_dir = save_dir / row[0]
            if not save_class_dir.is_dir():
                save_class_dir.mkdir(parents = True,
                                     exist_ok = True)
            
            save_file = save_class_dir / f"{idx}.png"
            cv2.imwrite(save_file, img)
            
            if idx%1000 == 0:
                print(f"[DATA] {idx} images have been saved into {save_dir}.")


extract_mnist(csv_fname = './mnist_train.csv',
              save_dir = './train')
    
    

