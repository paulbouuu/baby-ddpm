import torch
import torch.nn as nn



def beta_schedule(T, s=0.008):
    """ cosine noise schedule """

    t = torch.arange(T + 1, dtype=torch.float32)
    f_t = torch.cos(((t / T) + s) / (1 + s) * torch.pi / 2) ** 2

    alphas_cumprod = f_t[1:] / f_t[0]
    
    alphas = torch.empty_like(alphas_cumprod)
    alphas[0] = alphas_cumprod[0]
    alphas[1:] = alphas_cumprod[1:] / alphas_cumprod[:-1]

    betas = 1 - alphas

    return alphas, betas, alphas_cumprod

class Diffusion(nn.Module):
    def __init__(self, T=1000):
        super().__init__()
        self.T = T
        self.device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        # noise schedule
        alphas, betas, alphas_cumprod = beta_schedule(T=self.T)

        self.register_buffer("alphas", alphas)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)

        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))

    def q_sample(self, x0, t, noise=None):
        """ diffuse the data at timestep t """

        if noise is None:
            noise = torch.randn_like(x0) # (B, C, H, W)
        
        t = t.to(self.device)

        sqrt_ac = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        return sqrt_ac * x0 + sqrt_om * noise, noise

    def training_loss(self, model, x0):
        B, C, H, W = x0.shape
        t = torch.randint(0, self.T, (B,), device=self.device)
        
        # sample x_t and the added noise
        xt, noise = self.q_sample(x0, t)

        pred = model(xt, t) # prediction of the noise

        # MSE loss
        loss = (noise - pred).pow(2).mean()

        return loss
    
    @torch.no_grad()
    def p_sample(self, model, xt, t):
        """ one reverse denoising step: x_t -> x_{t-1} """

        t = t.to(self.device)

        beta_t = self.betas[t].view(-1, 1, 1, 1)
        alpha_t = self.alphas[t].view(-1, 1, 1, 1)
        alpha_cum_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)

        # noise prediction
        eps_theta = model(xt, t)

        # compute mean of q(x_{t-1} | x_t, x0)
        mean = (1 / torch.sqrt(alpha_t)) * (
            xt - (beta_t / torch.sqrt(1 - alpha_cum_t)) * eps_theta
        )

        # alpha_cum_{t-1}
        alpha_cum_prev = self.alphas_cumprod[(t - 1).clamp(min=0)].view(-1, 1, 1, 1)

        posterior_var = beta_t * (1 - alpha_cum_prev) / (1 - alpha_cum_t)
        posterior_std = torch.sqrt(posterior_var)

        # noise term (0 at t == 0)
        if (t == 0).all():
            return mean

        noise = torch.randn_like(xt)
        return mean + posterior_std * noise
    
    @torch.no_grad()
    def ddim_step(self, model, xt, t, t_prev):
        """ deterministic DDIM update: x_t -> x_{t_prev} """
        t = t.to(self.device)
        t_prev = t_prev.to(self.device)

        alpha_t = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        alpha_prev = self.alphas_cumprod[t_prev].view(-1, 1, 1, 1)

        eps = model(xt, t)

        # predict x0
        x0_pred = (xt - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)

        # deterministic DDIM update
        xt_prev = torch.sqrt(alpha_prev) * x0_pred + torch.sqrt(1 - alpha_prev) * eps

        return xt_prev # (B, C, H, W)
    
    @torch.no_grad()
    def sample_ddim(self, model, shape, steps=50):
        """ deterministic DDIM sampling with N steps instead of T (N << T) """

        B = shape[0]

        # start from pure noise
        xt = torch.randn(shape, device=self.device)

        # sequence of timesteps
        times = torch.linspace(self.T-1, 0, steps, dtype=torch.long, device=self.device)
    
        for i in range(len(times)):
            t = torch.full((B,), int(times[i]), dtype=torch.long, device=self.device)

            if i == len(times)-1:
                # last step predicts x0 directly
                t_prev = torch.zeros_like(t)
            else:
                t_prev = torch.full((B,), int(times[i+1]), dtype=torch.long, device=self.device)

            xt = self.ddim_step(model, xt, t, t_prev)

        return xt

