import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class IconDataset(Dataset):
    def __init__(self, root_dir, transform=None, sketch_transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.sketch_transform = sketch_transform
        self.filepairs = []

        # Find paired images (sketch and color icons)
        print("Finding image pairs...")
        for cls in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, cls)
            if os.path.isdir(class_dir):
                for subfolder in os.listdir(class_dir):
                    subfolder_dir = os.path.join(class_dir, subfolder)
                    if os.path.isdir(subfolder_dir):
                        sketch_path = None
                        color_icon_path = None

                        for file in os.listdir(subfolder_dir):
                            if "sketch_icon" in file:
                                sketch_path = os.path.join(subfolder_dir, file)
                            elif "color_icon" in file:
                                color_icon_path = os.path.join(subfolder_dir, file)

                        if sketch_path and color_icon_path:
                            self.filepairs.append((sketch_path, color_icon_path))

        print(f"Found {len(self.filepairs)} valid image pairs.")

    def __len__(self):
        return len(self.filepairs)

    def __getitem__(self, idx):
        sketch_icon_path, color_icon_path = self.filepairs[idx]

        # Load images
        sketch_icon = Image.open(sketch_icon_path).convert('L')  # Grayscale for sketches
        color_icon = Image.open(color_icon_path).convert('RGB')  # RGB for color icons

        # Apply transformations
        if self.sketch_transform:
            sketch_icon = self.sketch_transform(sketch_icon)
        if self.transform:
            color_icon = self.transform(color_icon)

        # Using a dummy label of 0
        label = 0
        return sketch_icon, color_icon, label


# Define image transformations
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # RGB normalization
])

sketch_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  # Grayscale normalization
])

# Load the dataset
root_dir = "D:\\RM\\processed_icons_extracted"
icon_dataset = IconDataset(root_dir=root_dir, transform=transform, sketch_transform=sketch_transform)

# Split the dataset into training, validation, and testing sets
def split_dataset(dataset, train_split=0.8, val_split=0.1):
    train_size = int(len(dataset) * train_split)
    val_size = int(len(dataset) * val_split)
    test_size = len(dataset) - train_size - val_size
    return torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

train_dataset, val_dataset, test_dataset = split_dataset(icon_dataset)

# Create DataLoaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

#Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4),  # Output a single channel
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        x = torch.mean(x, dim=[2, 3])  # Global Average Pooling to reduce spatial dimensions
        x = self.sigmoid(x)
        return x

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize models
generator = Generator().to(device)
discriminator = Discriminator().to(device)
criterion = nn.BCELoss()
optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
epochs = 10

# Directory to save models
save_dir = "D:\\RM"
os.makedirs(save_dir, exist_ok=True)

# Training loop with model saving
print("Starting training...")
for epoch in range(epochs):
    for i, (sketches, real_images, _) in enumerate(train_loader):
        sketches, real_images = sketches.to(device), real_images.to(device)
        batch_size = sketches.size(0)
        real_labels = torch.ones(batch_size, 1, device=device)
        fake_labels = torch.zeros(batch_size, 1, device=device)
        sketches = sketches.repeat(1, 3, 1, 1)  # Convert 1-channel sketches to 3-channel

        # Train Discriminator
        optimizer_D.zero_grad()
        fake_images = generator(sketches)
        real_outputs = discriminator(torch.cat((sketches, real_images), dim=1))  # Real input
        fake_outputs = discriminator(torch.cat((sketches, fake_images), dim=1))  # Fake input
        d_loss = criterion(real_outputs, real_labels) + criterion(fake_outputs, fake_labels)
        d_loss.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        fake_images = generator(sketches)
        outputs = discriminator(torch.cat((sketches, fake_images), dim=1))
        g_loss = criterion(outputs, real_labels)
        g_loss.backward()
        optimizer_G.step()

        if (i + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    # Save models at the end of each epoch
    torch.save(generator.state_dict(), os.path.join(save_dir, f"Icon_generator_epoch_{epoch+1}.pth"))
    torch.save(discriminator.state_dict(), os.path.join(save_dir, f"Icon_discriminator_epoch_{epoch+1}.pth"))
    print(f"Models saved for epoch {epoch+1}.")

print("Training complete.")