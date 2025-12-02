import torch
from torchvision.utils import save_image


def clear_cache(device):
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


class EMA:
    """ exponential moving average model """
    def __init__(self, model, beta=0.999):
        self.beta = beta
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self, model):
        with torch.no_grad():
            for k, v in model.state_dict().items():
                self.shadow[k] = self.beta * self.shadow[k] + (1 - self.beta) * v

    def copy_to(self, model):
        model.load_state_dict(self.shadow)


def compute_val_loss(model, diffusion, val_loader, device, ema=None):

    if ema is not None:
        raw_state = {k: v.clone() for k, v in model.state_dict().items()}
        ema.copy_to(model)

    model.eval()
    total = 0

    with torch.no_grad():
        for batch in val_loader:
            x = batch["image"].to(device)
            loss = diffusion.training_loss(model, x)
            total += loss.item()

    avg_loss = total / len(val_loader)

    if ema is not None:
        model.load_state_dict(raw_state)
    model.train()

    return avg_loss


def save_samples(epoch, model, diffusion, ema, sample_steps):
    # save raw model weights
    raw_state = {k: v.clone() for k, v in model.state_dict().items()}
    ema.copy_to(model)

    model.eval()
    with torch.no_grad():
        samples = diffusion.sample_ddim(
            model,
            (16, 3, 64, 64),
            steps=sample_steps
        )

    samples = (samples.clamp(-1, 1) + 1) / 2
    save_image(samples.cpu(), f"samples_epoch{epoch}.png", nrow=4)
    print(f"Saved samples for epoch {epoch}")

    # restore raw model weights
    model.load_state_dict(raw_state)
    model.train()
