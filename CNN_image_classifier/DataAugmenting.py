import os
from torchvision import datasets, transforms
from PIL import Image

# Define the path to the dataset
data_dir = 'Adataset/train'

# Define the transformations to apply to the images
transform = transforms.Compose([
    transforms.RandomChoice([
        transforms.RandomRotation(3),
        transforms.RandomRotation(1),
        transforms.RandomRotation(2)
    ]),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.mean(dim=0))
])

# Load the dataset and apply the transformations
dataset = datasets.ImageFolder(data_dir, transform=transform)
BaseDataset = datasets.ImageFolder(data_dir)

# Create the directory where the transformed images will be saved
os.makedirs('Adataset_transformed/train', exist_ok=True)

# Loop over the dataset and save the transformed images
for i, (image, label) in enumerate(dataset):
    # Get the class name from the label
    class_name = dataset.classes[label]

    # Create the directory for the class if it does not exist
    os.makedirs(f'Adataset_transformed/train/{class_name}', exist_ok=True)

    # Convert the tensor image to a PIL image
    image = transforms.ToPILImage()(image)

    # Save the image
    image.save(f'Adataset_transformed/train/{class_name}/image_{i}.png')
