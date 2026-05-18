import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------
# Vérification GPU
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device utilisé : {device}")

if torch.cuda.is_available():
    print("Nom du GPU :", torch.cuda.get_device_name(0))

# -----------------------
# Dataset aléatoire
# -----------------------
X = torch.randn(1000, 10).to(device)
y = torch.randn(1000, 1).to(device)

# -----------------------
# Modèle simple
# -----------------------
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
).to(device)

# -----------------------
# Loss + optimizer
# -----------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------
# Entraînement
# -----------------------
epochs = 20

for epoch in range(epochs):

    optimizer.zero_grad()

    outputs = model(X)

    loss = criterion(outputs, y)

    loss.backward()

    optimizer.step()

    print(f"Epoch {epoch+1}/{epochs} - Loss : {loss.item():.4f}")

print("Entraînement terminé.")