import torch
import torch.nn as nn
from custom_data import custom_dataset
import yaml
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from model import UNet
from noise_scheduler import linear_noise_scheduler
from engine import train_step
import numpy as np
from pathlib import Path

def train():
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Config File
    configPath1 = './default_b&w.yaml'
    configPath2 = './default_color.yaml'
    
    # dataPath
    dataPath1 = './train'
    dataPath2 = './textures'
    


    
    # Configurables / Flags / Hyperparameter
    showInfo = True
    loadModel = True
    
    
    configPath = configPath2
    dataPath = dataPath2
    modelLoadName = 'default_color1' #Where do you store the model you want to load
    saveName = "default_color1" #Where do you intend to save the training results
    
   
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0001
    NUM_WORKERS= 4
    EPOCHS = 100
    
    
    
    
    
    # Model File
    modelFile = Path(f'./result_{modelLoadName}')
    
    
    # Load config
    with open(configPath, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
    
    if showInfo:
        print(f"\n[INFO] The loaded config is as following: ")
        for key in config.keys():
            print(f"{key} : {config[key]}")
        print("\n")
    
    
    # 1. Load dataset
    trainData = custom_dataset(directory = dataPath)
    
    # 2. Visualize the dataset
    if showInfo:
        print(f"[INFO] The loaded dataset is as following: ")
        randNum = torch.randint(0, len(trainData)-1, (9,))
        for idx, num in enumerate(randNum):
            trainImg, trainLabel = trainData[num]
            
            trainImgPlt = ((trainImg + 1) / 2).permute(1,2,0)
            plt.subplot(3,3,idx+1)
            plt.imshow(trainImgPlt, cmap='gray')
            plt.title(f"Label: {trainLabel}")
            plt.axis(False)
        
        plt.tight_layout()
        plt.show()
            
        
        print(f"[INFO] The number of images in the dataset  : {len(trainData)}")
        print(f"[INFO] The size of an image                 : {trainImg.shape}")
        print(f"[INFO] The range of values in the image     : {trainImg.min():.4f} to {trainImg.max():.4f}")
        print(f"[INFO] The classes available in the dataset : {trainData.classes}")
        print("\n")
    
    
    # 3. Load the DataLoader
    trainDataLoader = DataLoader(dataset = trainData,
                                 batch_size = BATCH_SIZE,
                                 shuffle = True,
                                 num_workers = NUM_WORKERS)
    
    # 4. Visualize the DataLoader
    if showInfo:
        trainDataBatch, trainLabelBatch = next(iter(trainDataLoader))
        
        print(f"[INFO] The loaded dataloader is as following: ")
        print(f"[INFO] Number of batches in the dataloader  : {len(trainDataLoader)}")
        print(f"[INFO] Number of images per batch           : {trainDataBatch.shape[0]}")
        print(f"[INFO] Size of an image                     : {trainDataBatch[0].shape}")
    
    # 5. Create a model
    model0 = UNet(config = config,
                  device = device).to(device)
    
    # 6. Visualize the model
# =============================================================================
#     from torchinfo import summary
#     summary(model = model0,
#             input_size = (1,1,28,28),
#             col_names = ['input_size', 'output_size', 'num_params', 'trainable'],
#             row_settings = ['var_names'])
# =============================================================================

    # 7. Optimizer, lossFn, noise scheduler
    optimizer = torch.optim.Adam(params = model0.parameters(),
                                 lr = LEARNING_RATE)
    
    lossFn = nn.MSELoss()
    
    scheduler = linear_noise_scheduler(config = config,
                                       device = device)
    
    # 8. Create training loop
    
    # Initialize best Loss
    bestLoss = np.inf
    
    # Load Models
    if loadModel:
        model0.load_state_dict(torch.load(f = modelFile / 'best.pt',
                                          weights_only = True))
        model0 = model0.to(device)
        
        loadCKPT = torch.load(f = modelFile / 'ckpt.pt')
        
        bestLoss = loadCKPT['loss']
        
        optimizer.load_state_dict(loadCKPT['optimizer'])
        
        print("[CKPT] Loading the previously best trained model....")
        print(f"[CKPT] Epoch: {loadCKPT['epoch']}")
        print(f"[CKPT] Loss : {loadCKPT['loss']}")
    
    
    # Create save file for this training sessions
    resultFile = Path(f'./result_{saveName}')
    if not resultFile.is_dir():
        resultFile.mkdir(parents = True,
                         exist_ok = True)
    
    modelFile = resultFile / 'best.pt'
    ckptFile = resultFile / 'ckpt.pt'
    
    
    # Train
    model0.train()
    for epoch in range(EPOCHS):
    
        trainResult = train_step(model = model0,
                                 device = device,
                                 dataloader = trainDataLoader,
                                 optimizer = optimizer,
                                 loss_fn = lossFn,
                                 scheduler = scheduler,
                                 config = config)
        
        # Announce result for this epoch
        trainLoss = trainResult['loss']
        print(f"[INFO] Current Epoch: {epoch}")
        print(f"[INFO] Train Loss   : {trainLoss:.4f}")
        
        # Update CheckPoint
        if trainLoss < bestLoss:
            print(f"[CKPT] The best loss has been improved from {bestLoss} to {trainLoss}.")
            print(f"[CKPT] Proceed to save this as the best model at {modelFile}.")
            bestLoss = trainLoss
            
            torch.save(obj = model0.state_dict(),
                       f = modelFile)
            
            ckpt = {'epoch': epoch,
                    'loss': trainLoss,
                    'optimizer': optimizer.state_dict()}
            
            torch.save(obj = ckpt,
                       f = ckptFile)
            

if __name__ == '__main__':
    train()
    

