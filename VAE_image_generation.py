import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Update this to the exact name you used when saving the 6-layer model
# Make sure this matches the filename saved by your training script (e.g., "v7" or "v7_6layers")
version = "kle" 

# 1. Setup Device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# 2. Define the VAE Class (RESTORED TO 6 LAYERS)
class VAE(nn.Module):
    def __init__(self, latent_dim=1024): 
        super(VAE, self).__init__()
        
        # --- RESTORED: 6 layers including 1024 ---
        hidden_dims = [32, 64, 128, 256, 512, 1024]
        
        self.encoder_layers = nn.ModuleList()
        in_channels = 1 
        
        # --- RESTORED: Final Feature Map Size is 4x4 ---
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
        # --- RESTORED: Reshape to (Batch, 1024, 4, 4) ---
        result = result.view(-1, 1024, 4, 4)
        for layer in self.decoder_layers:
            result = layer(result)
        return self.final_layer(result)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

# 3. Load the Model
model = VAE(latent_dim=1024).to(device)

# Ensure this file exists in your folder!
model_path = f"vae_chest_xray_256_{version}.pth" 

try:
    # 'map_location' ensures it loads correctly on Mac/PC/Cloud
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Important: Sets model to evaluation mode
    print(f"Success: Loaded {model_path}!")
except FileNotFoundError:
    print(f"Error: Could not find file '{model_path}'. Check the version name.")
    exit()

# 4. Generate Images
print("Generating 5 random X-rays...")
with torch.no_grad():
    # Create 5 random latent vectors
    z = torch.randn(5, 1024).to(device)
    
    # Decode them into images
    generated_images = model.decode(z).cpu()
    
    # Plot them
    plt.figure(figsize=(15, 5))
    for i in range(5):
        ax = plt.subplot(1, 5, i + 1)
        img = generated_images[i].view(256, 256)
        plt.imshow(img, cmap='gray')
        plt.title(f"Gen #{i+1}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()