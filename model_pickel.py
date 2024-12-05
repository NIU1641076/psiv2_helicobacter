import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor
from sklearn.model_selection import train_test_split
from AEExample_Script import AEConfigs
from AEmodels import AutoEncoderCNN
import gc
import pickle
from PIL import Image  # Importa PIL.Image si no està ja importat

# Dataset personalitzat
class PreloadedImageDataset(torch.utils.data.Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Converteix la imatge a format PIL
        image = Image.fromarray((self.images[idx] * 255).astype('uint8'))  # Converteix de [0,1] a [0,255]
        if self.transform:
            image = self.transform(image)
        return image, 0  # Dummy label per compatibilitat amb DataLoader

# Ruta de les imatges pre-carregades
imatges_cropped_path = '/Users/guillemsallas/Desktop/Universitat/4t curs/1r Semestre/PSIV/Psiv_prj3/imatges_cropped_negatives.pkl' 
with open(imatges_cropped_path, 'rb') as f:
    imatges_cropped = pickle.load(f)

# Transformació per a les imatges
transform = Compose([
    Resize((256, 256)),
    ToTensor()
])

# Dividir les dades en entrenament i validació
train_images, val_images = train_test_split(imatges_cropped, test_size=0.2, random_state=42)
train_dataset = PreloadedImageDataset(train_images, transform=transform)
val_dataset = PreloadedImageDataset(val_images, transform=transform)

# Crear DataLoaders
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=16)
val_loader = DataLoader(val_dataset, shuffle=False, batch_size=16)

# Inicialitzar configuració i model
print("CREANT MODEL")
inputmodule_paramsEnc = {'num_input_channels': 3}
Config = "1"
net_paramsEnc, net_paramsDec, inputmodule_paramsDec = AEConfigs(Config, inputmodule_paramsEnc)
model = AutoEncoderCNN(inputmodule_paramsEnc, net_paramsEnc, inputmodule_paramsDec, net_paramsDec)

# Configuració del dispositiu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Optimitzador i funció de pèrdua
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
criterion = torch.nn.MSELoss()
print("MODEL CREAT")

# Entrenament i validació
num_epochs = 3
for epoch in range(num_epochs):
    # Modo entrenament
    model.train()
    running_loss = 0.0
    for inputs, _ in train_loader:
        inputs = inputs.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, inputs)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # Modo validació
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, inputs)

            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    # Imprimir resultats
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

# Guardar el model
torch.save(model.state_dict(), "./model_trained_4.pth")
print("Model guardat a model_trained_4.pth")

# Alliberar memòria
torch.cuda.empty_cache()
gc.collect()
