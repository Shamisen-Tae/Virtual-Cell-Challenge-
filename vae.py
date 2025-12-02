import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

class VAE(nn.Module):
    def __init__(self, n_genes, n_conditions, latent_dim=32):
        super(VAE, self).__init__()
        
        # Encoder
        self.enc1 = nn.Linear(n_genes + n_conditions, 256)
        self.enc2 = nn.Linear(256, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        
        # Decoder
        self.dec1 = nn.Linear(latent_dim + n_conditions, 128)
        self.dec2 = nn.Linear(128, 256)
        self.dec3 = nn.Linear(256, n_genes)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def encode(self, x, condition):
        h = torch.cat([x, condition], dim=1)
        h = self.dropout(self.relu(self.enc1(h)))
        h = self.dropout(self.relu(self.enc2(h)))
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, condition):
        h = torch.cat([z, condition], dim=1)
        h = self.dropout(self.relu(self.dec1(h)))
        h = self.dropout(self.relu(self.dec2(h)))
        return self.dec3(h)  # No activation - predicting log-space values
    
    def forward(self, x, condition):
        mu, logvar = self.encode(x, condition)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, condition), mu, logvar

def train_vae(X_train, y_train, latent_dim=64, epochs=100, batch_size=512,
                         learning_rate=1e-3, device='cuda'):
    
    n_genes = y_train.shape[1]
    n_conditions = X_train.shape[1]
    
    device = torch.device('cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    print(f"  Using device: {device}")
    
    model = VAE(n_genes, n_conditions, latent_dim).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
    
    X_train_torch = torch.FloatTensor(X_train).to(device)
    y_train_torch = torch.FloatTensor(y_train).to(device)
    
    train_dataset = TensorDataset(X_train_torch, y_train_torch)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        recon_losses = []
        kld_losses = []
        
        for condition, expression in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            recon, mu, logvar = model(expression, condition)
            
            # Loss: reconstruction + KL divergence
            recon_loss = nn.functional.mse_loss(recon, expression, reduction='mean')
            kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            
            # Start with very low beta, increase gradually
            beta = min(0.01, 0.001 * (epoch + 1) / 50)  # Gradually increase KL weight
            loss = recon_loss + beta * kld_loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * len(condition)
            recon_losses.append(recon_loss.item())
            kld_losses.append(kld_loss.item())
        
        avg_loss = train_loss / len(train_loader.dataset)
        avg_recon = np.mean(recon_losses)
        avg_kld = np.mean(kld_losses)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, "
                  f"Recon: {avg_recon:.4f}, KLD: {avg_kld:.4f}, Beta: {beta:.4f}")
        
        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 30:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        scheduler.step(avg_loss)
    
    return model


def predict_vae(model, X, device='cuda'):
    """
    Generate predictions: perturbation → gene expression
    """
    device = torch.device('cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    model.eval()
    
    X_torch = torch.FloatTensor(X).to(device)
    batch_size = X_torch.shape[0]
    
    with torch.no_grad():
        # Sample from prior
        z = torch.randn(batch_size, model.fc_mu.out_features).to(device)
        
        # Decode with condition
        predictions = model.decode(z, X_torch)
        
        return predictions.cpu().numpy()
