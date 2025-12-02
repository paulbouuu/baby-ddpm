import torch
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from utils import EMA
from model import UNet
from diffusion import Diffusion

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using device: mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using device: cuda")
else:
    device = torch.device("cpu")
    print("Using device: cpu")


# hyperparameters
batch_size = 32
lr = 1e-4
epochs = 20
image_size = 64
num_workers = 0
sample_steps = 50 # DDIM steps for sampling
diffusion_steps = 1000 # diffusion steps
rolling_avg_window = 100
# -----------------


# data
transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

dataset = load_dataset("zh-plus/tiny-imagenet", split="train")

def apply_transform(batch):
    batch["image"] = [transform(img) for img in batch["image"]]
    return batch

dataset.set_transform(apply_transform)

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=False
)

# model and diffusion
model = UNet(in_ch=3, out_ch=3, base=64, time_dim=128, diffusion_steps=diffusion_steps).to(device)
diffusion = Diffusion(T=diffusion_steps).to(device)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"{num_params/1e6:.2f}M parameters")

optimizer = optim.AdamW(model.parameters(), lr=lr)

ema = EMA(model, beta=0.999)

step = 0

losses = []
print(f"Rolling average window size: {rolling_avg_window}")

# training loop
for epoch in range(epochs):
    model.train()
    for batch in dataloader:
        x = batch["image"].to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            loss = diffusion.training_loss(model, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.update(model)

        losses.append(loss.item())
        rolling_avg = sum(losses[-rolling_avg_window:]) / min(len(losses), rolling_avg_window)

        if step % 200 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.6f}, Rolling avg loss: {rolling_avg:.6f}")

        step += 1

    # save model checkpoint
    torch.save({
        "ema_model": ema.shadow,
        "epoch": epoch}, 
        f"model_epoch{epoch}.pt"
    )

    print(f"Saved checkpoint epoch {epoch}")

    torch.mps.empty_cache()


print("Training complete!")
