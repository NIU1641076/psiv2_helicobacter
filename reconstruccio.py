import torch
import torchvision.transforms as transforms
from AEmodels import AutoEncoderCNN
from AEExample_Script import AEConfigs
import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Funció per reconstruir i guardar una imatge
def reconstruct_and_save_image(model, image_path, save_path, device):
    # Comprovar si la imatge existeix
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"La imatge no s'ha trobat: {image_path}")

    # Carregar i preprocessar la imatge
    input_image = cv2.imread(image_path)
    if input_image is None:
        raise ValueError(f"La imatge no s'ha pogut carregar correctament: {image_path}")
    
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = input_image.astype(np.float32) / 255.0
    input_image_pil = Image.fromarray((input_image * 255).astype(np.uint8))
    transform = transforms.Compose([transforms.ToTensor()])
    input_tensor = transform(input_image_pil).unsqueeze(0).to(device)

    # Inferència amb el model
    model.eval()
    with torch.no_grad():
        output_tensor = model(input_tensor)
    output_image = output_tensor.squeeze(0).cpu().numpy()
    output_image = np.transpose(output_image, (1, 2, 0))  # Convertir a format HWC

    # Guardar la reconstrucció com a imatge
    output_image_uint8 = (output_image * 255).astype(np.uint8)
    output_image_pil = Image.fromarray(output_image_uint8)
    output_image_pil.save(save_path)

    # Visualitzar l'original i la reconstrucció
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(input_image)
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(output_image)
    plt.title('Reconstructed Image')
    plt.show()

# Carregar el model entrenat
print("Carregant model...")
inputmodule_paramsEnc = {'num_input_channels': 3}
Config = "1"
net_paramsEnc, net_paramsDec, inputmodule_paramsDec = AEConfigs(Config, inputmodule_paramsEnc)
model = AutoEncoderCNN(inputmodule_paramsEnc, net_paramsEnc, inputmodule_paramsDec, net_paramsDec)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("./model_trained_2.pth", map_location=device))
model = model.to(device)
print("Model carregat.")

# Exemple de reconstrucció
test_image_path = "/Users/guillemsallas/Desktop/Universitat/4t curs/1r Semestre/PSIV/Psiv_prj3/Cropped/B22-193_0/5.png"
save_path = "./reconstructed_output.png"
reconstruct_and_save_image(model, test_image_path, save_path, device)
print(f"Reconstrucció guardada a {save_path}")
