import torch


class linear_noise_scheduler():

    def __init__(self, config, device):
        
        self.beta_start = config['beta_start']
        self.beta_end = config['beta_end']
        self.num_timesteps = config['num_timesteps']
        
        self.beta = torch.linspace(self.beta_start, self.beta_end, self.num_timesteps).to(device) # 10000
        self.alpha = 1 - self.beta # 1000
        self.alpha_hat = torch.cumprod(self.alpha, dim=0) #1000
        
        

    def add_noise(self, x, noise, t):
        """
            x: Original image, cuda
            noise: Noise, same shape as x, cuda
            t: time step, torch tensor, (B)
        """
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[..., None, None, None]
        sqrt_one_minus_alpha_hat  = torch.sqrt(1 - self.alpha_hat[t])[..., None, None, None]

        
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise
    
    def sample_prev(self, x, pred_noise, t):
        """
            x: Original image, cuda
            pred_noise: Noise, same shape as x, cuda
            t: time step, int, torch tensor
        """
        
        x0 = torch.sqrt((1-self.alpha_hat[t])) * pred_noise
        x0 = x - x0
        x0 = x0 / torch.sqrt(self.alpha_hat[t])
        
        mean = (1 - self.alpha[t]) / (torch.sqrt(1 - self.alpha_hat[t])) * pred_noise
        mean = x - mean
        mean = mean / torch.sqrt(self.alpha[t])
        
        if t==0:
            return mean, x0
        
        else:    
            variance = (1 - self.alpha[t]) * (1 - self.alpha_hat[t-1])
            variance = variance / (1 - self.alpha_hat[t])
            std_dev = variance ** 0.5
            z = torch.randn_like(mean)
            
            return mean + std_dev*z, x0
