import os
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.io import decode_image


class AutonomousCarDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):                  # La fonction __len__ renvoie le nombre d'échantillons dans notre ensemble de données.
        return len(self.data)

    

    def __getitem__(self, idx):         # La fonction __getitem__ charge et renvoie un échantillon du jeu de données à l'index donné 
        row = self.data.iloc[idx]

        # Le timestamp sert d'identifiant d'image
        timestamp = str(int(float(str(row["time_in_ms"]).replace(",", "."))))

        image_path = os.path.join(self.image_dir, timestamp + ".jpg")
        image = Image.open(image_path).convert("RGB")

        left_command = self.gpio_coeff_to_signed_command(
            row["gpio_left"],
            row["coeff_left"]
        )

        right_command = self.gpio_coeff_to_signed_command(
            row["gpio_right"],
            row["coeff_right"]
        )

        target = torch.tensor(
            [left_command, right_command],
            dtype=torch.float32
        )

        if self.transform:
            image = self.transform(image)

        return image, target
    

transform = transforms.Compose([
    transforms.Resize((120, 160)),
    transforms.ToTensor()
])

dataset = AutonomousCarDataset ( 
    csv_file = "dataSet/#Record_2025-07-02_09-46-48/labels.csv",
    image_dir = "dataSet/#Record_2025-07-02_09-46-48/Images",
    transform = transform
)

loader = DataLoader(dataset, batch_size = 8, shuffle = True )

for images, targets in loader:
    print("Images :", images.shape)
    print("Targets :", targets.shape)
    print("Premières commandes :", targets[:5])
    break