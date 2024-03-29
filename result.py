from utils import load_model
from torchvision.models import alexnet
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import torchvision.transforms as transforms
import torch
from PIL import Image
import torch.nn.functional as F
import json

filename = 'alexnet_weights_best_acc.tar' # pre-trained model path
use_gpu = False # load weights on the gpu
model = alexnet(num_classes=1081) # 1081 classes in Pl@ntNet-300K

load_model(model, filename=filename, use_gpu=use_gpu)

def preprocess_image(image_path):
    preprocess = transforms.Compose([
        transforms.Resize((139, 139)),  # Resize to match model input size
        transforms.ToTensor(),           # Convert PIL image to tensor
    ])
    image = Image.open(image_path).convert('RGB')
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

model.eval()
with torch.no_grad():
    output = model(preprocess_image("lactuca.webp"))

probabilities = F.softmax(output, dim=1)

probabilities_list = probabilities[0].tolist()

max_probabilities = max(probabilities_list)

pick = probabilities_list.index(max_probabilities)


with open('plantnet_300K/plantnet300K_species_id_2_name.json') as f:
    data = json.load(f)
data_list = []

for item in data:
    data_list.append(item)
print(pick)
print(data_list[pick])

