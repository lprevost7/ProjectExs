import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from cnn import Net

# Charger le modèle
PATH = './model/best_model_fromscratch'

model = Net()
model.load_state_dict(torch.load(PATH))
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Définir la transformation
transform = transforms.Compose([
    transforms.RandomResizedCrop(300,scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

# Liste des noms de classes (à remplacer par vos propres noms de classes)
classes = ('Cat', 'Dog')

# Liste des chemins d'images à tester
image_paths = ['testimages/0.jpg','testimages/had.jpg', 'testimages/both.jpg', 'testimages/177546.jpg', 'testimages/top.jpg', 'testimages/insta.jpg', 'testimages/ver.jpg', 'testimages/t.jpg',]

for image_path in image_paths:
    # Charger l'image
    image = Image.open(image_path)

    # Transformer l'image en tenseur et ajouter une dimension supplémentaire
    image_tensor = transform(image).unsqueeze(0)
    image_tensor = image_tensor.to(device)

    # Effectuer une prédiction
    output = model(image_tensor)
    _, predicted = torch.max(output, 1)
    score = torch.nn.functional.softmax(output, dim=1)[0][predicted].item()

    # Afficher l'image et le score de classification
    plt.imshow(image)
    plt.title(f'Classe prédite: {classes[predicted]}, Score: {score:.2f}')
    plt.show()
