from tqdm import tqdm
import torch


def train_step(model, device, dataloader, optimizer, loss_fn, scheduler, config):
    
    num_timesteps = config['num_timesteps']
    train_loss = 0
    
    
    for batch, (X,_) in enumerate(tqdm(dataloader, position=0)):
        
        B, _, _, _ = X.shape
        
        X = X.to(device)
        timestep = torch.randint(0, num_timesteps, (B,)).to(device)
        noise = torch.randn_like(X)
        
        X = scheduler.add_noise(x = X,
                                noise = noise,
                                t = timestep)
        
        pred_noise = model(x = X,
                           t = timestep)
        
        loss = loss_fn(pred_noise, noise)
        train_loss += loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    train_loss /= len(dataloader)
    
    return {'model_name': model.__class__.__name__,
            'loss': train_loss}
        
        

