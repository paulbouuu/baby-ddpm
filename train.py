import torch
import torch.optim as optim
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms, utils

from utils import EMA
from model import UNet
from diffusion import Diffusion


# hyperparameters
batch_size = 8
lr = 1e-4
epochs = 20
image_size = 64
num_workers = 4
save_every = 500
sample_steps = 50 # DDIM steps for sampling
# -----------------


# data
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

dataset = load_dataset("zh-plus/tiny-imagenet", split="train")

def apply_transform(sample):
    sample["image"] = transform(sample["image"])
    return sample

dataset.set_transform(apply_transform)

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True
)

# model and diffusion
model = UNet(in_ch=3, out_ch=3, base=64, time_dim=128)
diffusion = Diffusion(T=1000)

optimizer = optim.AdamW(model.parameters(), lr=lr)

ema = EMA(model)

step = 0

# training loop
for epoch in range(epochs):
    for x, _ in dataloader:

        loss = diffusion.training_loss(model, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.update(model)

        if step % 100 == 0:
            print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")

        # save samples
        if step % save_every == 0:
            model.eval()
            ema.copy_to(model)

            with torch.no_grad():
                samples = diffusion.sample_ddim(model, shape=(8, 3, image_size, image_size), steps=sample_steps)
                samples = samples.clamp(-1, 1)
                samples = (samples + 1) * 0.5

            utils.save_image(samples, f"samples_{step}.png", nrow=4)
            print(f"Saved sample at step {step}")

            model.train()

        step += 1

    # save model checkpoint
    torch.save(model.state_dict(), f"model_epoch{epoch}.pt")
    print(f"Saved checkpoint epoch {epoch}")

print("Training complete!")
