import torch
import torch.nn as nn

def get_sinusoidal_embedding(timestep, time_embedding_dim, device):
    
    """
        timestep: torch tensor, integer, (B), device = 'cuda'
        time_embedding_dim: integer, (1)
    """
    
    assert time_embedding_dim % 2 == 0, f"[ERROR] Provided time embedding_dim is not divisible by 2."
    
    timestep = timestep[..., None].repeat(1, time_embedding_dim // 2) # B x (time_embedding_dim/2)
    
    factor = (torch.arange(0, time_embedding_dim//2) * 2) / time_embedding_dim # B x (time_embedding_dim/2)
    factor = 10000 ** factor
    factor = factor.to(device)
    
    component = timestep / factor # B  x (time_embedding_dim / 2)
    sin_comp = torch.sin(component) # B  x (time_embedding_dim / 2)
    cos_comp = torch.cos(component) # B  x (time_embedding_dim / 2)
    
    result = torch.cat([sin_comp, cos_comp], dim=1) # B x time_embedding_dim
    
    return result


class down_block(nn.Module):
    
    def __init__(self, config, in_channels, out_channels, down_sample):
        super().__init__()
        
        self.num_down_layers = config['num_down_layers']
        self.time_embedding_dim = config['time_embedding_dim']
        self.num_heads = config['num_heads']
        
        self.first_resnet_conv = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(num_groups = 8,
                             num_channels = in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels = in_channels if i == 0 else out_channels,
                          out_channels = out_channels,
                          kernel_size = 3,
                          stride = 1,
                          padding = 1)
                ) for i in range(self.num_down_layers)])
        
        self.time_embedding_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(in_features = self.time_embedding_dim,
                          out_features = out_channels)
                )for _ in range(self.num_down_layers)])
        
        self.second_resnet_conv = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(num_groups = 8,
                             num_channels = out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels = out_channels,
                          out_channels = out_channels,
                          kernel_size = 3,
                          stride = 1,
                          padding = 1)
                ) for _ in range(self.num_down_layers)])
        
        self.attn_norm = nn.ModuleList([
            nn.GroupNorm(num_groups = 8,
                         num_channels = out_channels)
            for _ in range(self.num_down_layers)])
        
        self.attn_block = nn.ModuleList([
            nn.MultiheadAttention(embed_dim = out_channels,
                                  num_heads = self.num_heads,
                                  batch_first = True)
            for _ in range(self.num_down_layers)])
    
        self.residual_input_conv = nn.ModuleList([
            nn.Conv2d(in_channels = in_channels if i == 0 else out_channels,
                      out_channels = out_channels,
                      kernel_size = 1,
                      stride = 1,
                      padding = 0)
            for i in range(self.num_down_layers)])
        
        self.down_sample_conv = nn.Conv2d(in_channels = out_channels,
                                          out_channels = out_channels,
                                          kernel_size = 4,
                                          stride = 2,
                                          padding = 1) if down_sample else nn.Identity()
    
    def forward(self, x, t_emb):
        
        out = x
         
        for i in range(self.num_down_layers):
            
            residual_inpt = out
            
            out = self.first_resnet_conv[i](out)
            out = out + self.time_embedding_layers[i](t_emb)[..., None, None]
            out = self.second_resnet_conv[i](out)
            out = out + self.residual_input_conv[i](residual_inpt)
            
            B, C, H, W = out.shape
            attn_in = out
            
            out = out.reshape((B, C, H*W))
            out = self.attn_norm[i](out)
            out = out.transpose(1,2)
            attn_out, _ = self.attn_block[i](out, out, out)
            attn_out = attn_out.transpose(1,2)
            attn_out = attn_out.reshape((B, C, H, W))
            out = attn_out + attn_in
        
        out = self.down_sample_conv(out)    
        return out



class mid_block(nn.Module):
    
    def __init__(self, config, in_channels, out_channels):
        super().__init__()
        
        self.num_mid_layers = config['num_mid_layers']
        self.time_embedding_dim = config['time_embedding_dim']
        self.num_heads = config['num_heads']
        
        self.first_resnet_conv = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(num_groups = 8,
                             num_channels = in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels = in_channels if i == 0 else out_channels,
                          out_channels = out_channels,
                          kernel_size = 3,
                          stride = 1,
                          padding = 1)
                ) for i in range(self.num_mid_layers + 1)])
        
        self.time_embedding_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(in_features = self.time_embedding_dim,
                          out_features = out_channels)
                ) for _ in range(self.num_mid_layers + 1)])
        
        self.second_resnet_conv = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(num_groups = 8,
                             num_channels = out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels = out_channels,
                          out_channels = out_channels,
                          kernel_size = 3,
                          stride = 1,
                          padding = 1)
                ) for _ in range(self.num_mid_layers + 1)])
        
        self.attn_norm = nn.ModuleList([
            nn.GroupNorm(num_groups = 8,
                         num_channels = out_channels)
            for _ in range(self.num_mid_layers)])
        
        self.attn_block = nn.ModuleList([
            nn.MultiheadAttention(embed_dim = out_channels,
                                  num_heads = self.num_heads,
                                  batch_first = True)
            for _ in range(self.num_mid_layers)])
        
        self.residual_input_conv = nn.ModuleList([
            nn.Conv2d(in_channels = in_channels if i==0 else out_channels,
                      out_channels = out_channels,
                      kernel_size = 1,
                      stride = 1,
                      padding = 0)
            for i in range(self.num_mid_layers + 1)])
    
    def forward(self, x, t_emb):
        
        out = x
        residual_inpt = out
        
        out = self.first_resnet_conv[0](out)
        out = out + self.time_embedding_layers[0](t_emb)[..., None, None]
        out = self.second_resnet_conv[0](out)
        out = out + self.residual_input_conv[0](residual_inpt)
        
        for i in range(self.num_mid_layers):
            
            B, C, H, W = out.shape
            attn_in = out
            
            out = out.reshape((B, C, (H*W)))
            out = self.attn_norm[i](out)
            out = out.transpose(1,2)
            attn_out, _ = self.attn_block[i](out, out, out)
            attn_out = attn_out.transpose(1,2)
            attn_out = attn_out.reshape((B, C, H, W))
            out = attn_out + attn_in
            
            residual_inpt = out
            
            out = self.first_resnet_conv[i+1](out)
            out = out + self.time_embedding_layers[i+1](t_emb)[..., None, None]
            out = self.second_resnet_conv[i+1](out)
            out = out + self.residual_input_conv[i+1](residual_inpt)
        
        return out


class up_block(nn.Module):
    
    def __init__(self, config, in_channels, out_channels, up_sample):
        super().__init__()
        
        self.num_up_layers = config['num_up_layers']
        self.time_embedding_dim = config['time_embedding_dim']
        self.num_heads = config['num_heads']
        
        self.first_resnet_conv = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(num_groups = 8,
                             num_channels = in_channels if i == 0 else out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels = in_channels if i == 0 else out_channels,
                          out_channels = out_channels,
                          kernel_size = 3,
                          stride = 1,
                          padding = 1)
                ) for i in range(self.num_up_layers)])
        
        self.time_embedding_layers = nn.ModuleList([
            nn.Sequential(
                nn.SiLU(),
                nn.Linear(in_features = self.time_embedding_dim,
                          out_features = out_channels)
                ) for _ in range(self.num_up_layers)])
        
        self.second_resnet_conv = nn.ModuleList([
            nn.Sequential(
                nn.GroupNorm(num_groups = 8,
                             num_channels = out_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels = out_channels,
                          out_channels = out_channels,
                          kernel_size = 3,
                          stride = 1,
                          padding = 1)
                ) for _ in range(self.num_up_layers)])
        
        self.attn_norm = nn.ModuleList([
            nn.GroupNorm(num_groups = 8,
                         num_channels = out_channels)
            for _ in range(self.num_up_layers)])
        
        self.attn_block = nn.ModuleList([
            nn.MultiheadAttention(embed_dim = out_channels,
                                  num_heads = self.num_heads,
                                  batch_first = True)
            for _ in range(self.num_up_layers)])
        
        self.residual_input_conv = nn.ModuleList([
            nn.Conv2d(in_channels = in_channels if i ==0 else out_channels,
                      out_channels = out_channels,
                      kernel_size = 1,
                      stride = 1,
                      padding = 0)
            for i in range(self.num_up_layers)])
        
        self.up_sample_conv = nn.ConvTranspose2d(in_channels = in_channels // 2,
                                                 out_channels = in_channels // 2,
                                                 kernel_size = 4,
                                                 stride = 2,
                                                 padding = 1) if up_sample else nn.Identity()
        
    
    def forward(self, x, t_emb, down_map):
        
        out = x
        out = self.up_sample_conv(out)
        out = torch.cat([out, down_map], dim=1)
        
        for i in range(self.num_up_layers):
            
            residual_inpt = out
            
            out = self.first_resnet_conv[i](out)
            out = out + self.time_embedding_layers[i](t_emb)[..., None, None]
            out = self.second_resnet_conv[i](out)
            out = out + self.residual_input_conv[i](residual_inpt)
            
            B, C, H, W = out.shape
            attn_in = out
            
            out = out.reshape((B, C, (H*W)))
            out = self.attn_norm[i](out)
            out = out.transpose(1,2)
            attn_out, _ = self.attn_block[i](out, out, out)
            attn_out = attn_out.transpose(1,2)
            attn_out = attn_out.reshape((B, C, H, W))
            out = attn_out + attn_in
            
            return out
    

class UNet(nn.Module):
    
    def __init__(self, config, device):
        super().__init__()
        self.down_channels = config['down_channels']
        self.mid_channels = config['mid_channels']
        self.up_channels = list(reversed(self.down_channels))
        
        self.down_samples = config['down_samples']
        self.up_samples = list(reversed(self.down_samples))
        
        self.im_channels = config['im_channels']
        self.time_embedding_dim = config['time_embedding_dim']
        self.device = device
        
        assert self.down_channels[-1] == self.mid_channels[0], "[ERROR] Last down channel does not match with the first mid channel"
        assert self.down_channels[-2] == self.mid_channels[-1], "[ERROR] Last second down channel does not match with the last mid channel"
        assert len(self.down_channels) - 1 == len(self.down_samples), "[ERROR] Number of down channels does not match with number of down samples"
        
        
        self.time_embedding_block = nn.Sequential(nn.Linear(in_features = self.time_embedding_dim,
                                                            out_features = self.time_embedding_dim),
                                                  nn.SiLU(),
                                                  nn.Linear(in_features = self.time_embedding_dim,
                                                            out_features = self.time_embedding_dim))
        
        self.inpt_conv = nn.Conv2d(in_channels = self.im_channels,
                                   out_channels = self.down_channels[0],
                                   kernel_size = 3,
                                   stride = 1,
                                   padding = 1)
        
        
        self.DOWN_BLOCKS = nn.ModuleList([])
        for i in range(len(self.down_channels) - 1):
            self.DOWN_BLOCKS.append(down_block(config = config,
                                               in_channels = self.down_channels[i],
                                               out_channels = self.down_channels[i+1],
                                               down_sample = self.down_samples[i]))
        
        self.MID_BLOCKS = nn.ModuleList([])
        for i in range(len(self.mid_channels) - 1):
            self.MID_BLOCKS.append(mid_block(config = config,
                                             in_channels = self.mid_channels[i],
                                             out_channels = self.mid_channels[i+1]))
                    
        self.UP_BLOCKS = nn.ModuleList([])
        for i in range(len(self.up_channels) - 1):
            self.UP_BLOCKS.append(up_block(config = config,
                                           in_channels = self.up_channels[i+1] * 2,
                                           out_channels = 16 if i == (len(self.up_channels) - 2) else self.up_channels[i+2],
                                           up_sample = self.up_samples[i]))
        
        self.outpt_conv = nn.Sequential(nn.GroupNorm(num_groups = 8,
                                                     num_channels = 16),
                                        nn.SiLU(),
                                        nn.Conv2d(in_channels = 16,
                                                  out_channels = self.im_channels,
                                                  kernel_size = 3,
                                                  stride = 1,
                                                  padding = 1))
    
    def forward(self, x, t = None):
        
        """
            x: torch tensor, float, cuda,
            t: torch tensor, (B,), float, cuda
        """
        
        if t == None:
            print("[INFO] Entering testing mode... ...")
            t = torch.tensor([0]).to(self.device)
        
        t_emb = get_sinusoidal_embedding(timestep = t, 
                                         time_embedding_dim = self.time_embedding_dim, 
                                         device = self.device) # B x time_embedding_dim, cuda
        
        t_emb = self.time_embedding_block(t_emb)
        
        out = x 
        out = self.inpt_conv(out)
        
        down_maps = []
        for block in self.DOWN_BLOCKS:
            down_maps.append(out)
            out = block(x = out,
                        t_emb = t_emb)
        
        for block in self.MID_BLOCKS:
            out = block(x = out,
                        t_emb = t_emb)
        
        for block in self.UP_BLOCKS:
            down_map = down_maps.pop()
            out = block(x = out,
                        t_emb = t_emb,
                        down_map = down_map)
        
        out = self.outpt_conv(out)
        return out
        
        
        
        
            
            
            
        
        
        
        
    
            
        

