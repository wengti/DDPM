import torch
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
import cv2
from pathlib import Path
import yaml
from model import UNet
from noise_scheduler import linear_noise_scheduler
from tqdm import tqdm

def sample(model, scheduler, config, device, save_folder):
    
    num_timesteps = config['num_timesteps']
    num_samples = config['num_samples']
    num_grid_rows = config['num_grid_rows']
    im_channels = config['im_channels']
    
    save_folder = Path(save_folder)
    if not save_folder.is_dir():
        save_folder.mkdir(parents = True,
                          exist_ok = True)
    
    save_dir = save_folder / 'Generated Images'
    if not save_dir.is_dir():
        save_dir.mkdir(parents = True,
                       exist_ok = True)
    
    out = torch.randn((num_samples, im_channels, 28, 28)).to(device)

    
    model.eval()
    with torch.inference_mode():
        
        for t in tqdm(list(reversed(range(num_timesteps))), position=0):
            
            timestep = torch.tensor([t]).to(device)
            
            pred_noise = model(x = out,
                               t = timestep.repeat(num_samples))
            
            out, x0 = scheduler.sample_prev(x = out,
                                            pred_noise = pred_noise,
                                            t = timestep)
            
            output = torch.clamp(out, -1, 1)
            x0_out = torch.clamp(x0, -1, 1)
            
            output = ((output + 1) / 2).detach().cpu()
            x0_out = ((x0_out + 1) / 2).detach().cpu()
            
            grid = make_grid(tensor = output,
                             nrow = num_grid_rows)
            out_image = ToPILImage()(grid)

            save_file = save_dir / f"{t}.png"
            out_image.save(save_file)


def infer():
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Variables
    configPath1 = './default_b&w.yaml'
    configPath2 = './default_color.yaml'
    
    # Configurables
    saveName = 'default_color1'
    configPath = configPath2
    
    
    
    
    
    

    # Save File
    saveFile = Path(f'./result_{saveName}')
    if not saveFile.is_dir():
        saveFile.mkdir(parents = True,
                       exist_ok = True)
    
    # Read config
    with open(configPath, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
    
    # Load model
    model0 = UNet(config = config,
                  device = device).to(device)
    
    model0.load_state_dict(torch.load(f = saveFile/'best.pt',
                                      weights_only = True))
    model0 = model0.to(device)
    
    # Create scheduler
    scheduler = linear_noise_scheduler(config = config,
                                       device = device)
    
    # Sample
    sample(model = model0,
           scheduler = scheduler,
           config = config,
           device = device,
           save_folder = saveFile)

if __name__ == '__main__':
    infer()
    
            

            
        
        
    

