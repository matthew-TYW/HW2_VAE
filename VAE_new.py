# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from ignite.metrics import FID, InceptionScore

#==============================
#=========Parameters===========
#==============================
version= "kle" # For saving/loading models with different configurations
lr= 0.0005
epochs= 35
batch_size= 32
#==============================

# 1. Setup Device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Success: Using Apple Mac GPU!")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Success: Using NVIDIA GPU (CUDA)!")
else:
    device = torch.device("cpu")
    print("Warning: Using CPU.")

# 2. Define Metrics
fid_metric = FID(device=device)
is_metric = InceptionScore(device=device, output_transform=lambda x: x[0])

# 3. Helper Function for Inception
def prepare_for_inception(x):
    x = (x + 1) / 2
    x = torch.clamp(x, 0, 1)
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1) 
    x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
    return x

# 4. Define the VAE Class
class VAE(nn.Module):
    def __init__(self, latent_dim=1024): 
        super(VAE, self).__init__()
        
        # --- 6 layers  ---
        hidden_dims = [32, 64, 128, 256, 512, 1024]
        self.encoder_layers = nn.ModuleList()
        in_channels = 1 
        
        # 256 / (2^6) = 4
        final_size = 4 
        final_px = final_size * final_size # 16
        
        for h_dim in hidden_dims:
            self.encoder_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim

        # Latent Vectors
        self.fc_mu = nn.Linear(hidden_dims[-1] * final_px, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1] * final_px, latent_dim)
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * final_px)

        # Decoder
        hidden_dims.reverse()
        self.decoder_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i], hidden_dims[i+1],
                                       kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.LeakyReLU()
                )
            )

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1], 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh() 
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x):
        result = x
        for layer in self.encoder_layers:
            result = layer(result)
        result = torch.flatten(result, start_dim=1)
        return self.fc_mu(result), self.fc_var(result)

    def decode(self, z):
        result = self.decoder_input(z)
        result = result.view(-1, 1024, 4, 4)
        for layer in self.decoder_layers:
            result = layer(result)
        return self.final_layer(result)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

# 5. Initialization
model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr) 

def loss_function(recon_x, x, mu, log_var):
    recons_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return recons_loss + kld_loss

# 6. Data Loading
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)) 
])

dataset = datasets.ImageFolder(root='./chest_xray/train', transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# %% Training Loop

train_losses = []
fid_history = []
is_history = []

print(f"Starting Training ({len(dataset)} images)...")

for epoch in range(epochs):
    # --- TRAINING ---
    model.train()
    total_epoch_loss = 0
    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        
        recon_batch, mu, log_var = model(data)
        loss = loss_function(recon_batch, data, mu, log_var)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_epoch_loss += loss.item()

    # --- EVALUATION ---
    model.eval()
    fid_metric.reset()
    is_metric.reset()
    
    with torch.no_grad():
        for i, (real_imgs, _) in enumerate(dataloader):
            if i > 5: break 
            real_imgs = real_imgs.to(device)
            z = torch.randn(real_imgs.size(0), 1024).to(device)
            fake_imgs = model.decode(z)
            
            real_inc = prepare_for_inception(real_imgs)
            fake_inc = prepare_for_inception(fake_imgs)
            
            fid_metric.update((fake_inc, real_inc))
            is_metric.update(fake_inc)

    avg_loss = total_epoch_loss / len(dataloader.dataset)
    epoch_fid = fid_metric.compute()
    epoch_is = is_metric.compute()
    
    train_losses.append(avg_loss)
    fid_history.append(epoch_fid)
    is_history.append(epoch_is)
    
    print(f'Epoch: {epoch+1} | Loss: {avg_loss:.1f} | FID: {epoch_fid:.4f} | IS: {epoch_is:.4f}')

# %% --- POST TRAINING PLOTS & SAVING ---

save_path = f"vae_chest_xray_256_{version}.pth"
torch.save(model.state_dict(), save_path)
print(f"Model saved to {save_path}")

x_axis = range(1, epochs + 1)
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

axs[0].plot(x_axis, train_losses, color='blue', label='Training Loss')
axs[0].set_title("Training Loss")
axs[0].grid(True)

axs[1].plot(x_axis, fid_history, color='red', label='FID Score')
axs[1].set_title("FID Score")
axs[1].grid(True)

axs[2].plot(x_axis, is_history, color='green', label='Inception Score')
axs[2].set_title("Inception Score")
axs[2].grid(True)

plt.tight_layout()
plt.show()

# %% --- VISUALIZE GENERATED IMAGES ---

print("Generating images...")
model.eval()

# 1. Generate New AI X-Ray
with torch.no_grad():
    z = torch.randn(1, 1024).to(device)
    generated_image = model.decode(z).cpu().view(256, 256)
    
    plt.figure(figsize=(4, 4))
    plt.imshow(generated_image, cmap='gray')
    plt.title("VAE Generated Chest X-Ray")
    plt.axis('off')
    plt.show()

# 2. Reconstruct Existing X-Ray
with torch.no_grad():
    real_img, _ = next(iter(dataloader))
    real_img = real_img[0].unsqueeze(0).to(device)
    recon_img, _, _ = model(real_img)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(real_img.cpu().view(256, 256), cmap='gray')
    axes[0].set_title("Original")
    axes[0].axis('off')
    axes[1].imshow(recon_img.cpu().view(256, 256), cmap='gray')
    axes[1].set_title("Reconstructed")
    axes[1].axis('off')
    plt.show()