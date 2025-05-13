import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import tqdm
import time


# CIFAR-10 label names
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

class CustomDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data.iloc[idx]
        image_path = sample['image_path']
        label = sample['label']

        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

def collate_fn(batch):
    # Разделение батча на признаки и метки
    features, labels = zip(*batch)

    # Преобразование признаков в тензоры
    # Если признаки - это изображения или многомерные данные, то используем torch.stack
    # В случае с изображениями, например, мы просто соединяем тензоры вдоль нового измерения.
    features = torch.stack([torch.tensor(f) for f in features])

    # Если необходимо паддинг признаков до одинаковой длины
    # Например, если это последовательности, то их нужно паддировать до максимальной длины
    if isinstance(features[0], torch.Tensor) and features[0].dim() > 1:
        # Например, если features - это изображения, где у нас размерность (C, H, W), то можно сделать padding для каждого изображения
        max_height = max(f.size(1) for f in features)
        max_width = max(f.size(2) for f in features)
        
        # Паддинг по ширине и высоте
        padded_features = []
        for f in features:
            padded_image = torch.nn.functional.pad(f, (0, max_width - f.size(2), 0, max_height - f.size(1)))
            padded_features.append(padded_image)
        features = torch.stack(padded_features)
    
    # Преобразование меток в тензор
    labels = torch.tensor(labels, dtype=torch.long)

    return features, labels

def get_cifar():
    # Загрузка данных CIFAR-10
    from torchvision.datasets import CIFAR10
    dataset = CIFAR10(root='./data', train=True, download=True)
    # Создание директории для сохранения изображений
    os.makedirs('./data/cifar10_images', exist_ok=True)

    # Сохранение изображений и создание CSV файла
    data = []
    for idx, (image, label) in enumerate(tqdm.tqdm(dataset)):
        # Сохранение изображения в формате PNG
        image_path = f'./data/cifar10_images/{idx}.png'
        image.save(image_path)
        data.append([image_path, label])

    # Сохранение данных в CSV файл
    df = pd.DataFrame(data, columns=['image_path', 'label'])
    df.to_csv('./data/cifar10.csv', index=False)


from torchvision import transforms
from torch.utils.data import DataLoader

def fake_training_loop(dataloader, device='cpu', folder = './'):
    os.makedirs(folder, exist_ok=True)
    
    for i, (images, labels) in enumerate(tqdm.tqdm(dataloader)):
        images = images.to(device)

        # upsample for visibility (32x32 -> 128x128)
        images = torch.nn.functional.interpolate(images, scale_factor=4, mode='bicubic')

        if i % 1000 == 0:
            # Create a horizontal grid of images
            concat = torch.cat([img for img in images], dim=2)  # Concatenate along width
            # Convert to [H, W, C] and uint8 for PIL
            concat = concat.mul(255).clamp(0, 255).byte().cpu().permute(1, 2, 0).numpy()
            # Расшифровываем метки
            decoded_labels = [CIFAR10_CLASSES[label] for label in labels]
            print(f"Batch {i} labels:", decoded_labels)
            # Сохраняем на диск
            filename = f"batch_{i}.png"
            filename = os.path.join(folder, filename)
            
            Image.fromarray(concat).save(filename)  

if __name__ == '__main__':
    if not os.path.exists('./data/cifar10_images'):
        get_cifar()

    # Определение аугментаций
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor()
    ])

    # Создание датасета и DataLoader
    csv_file = './data/cifar10.csv'
    dataset = CustomDataset(csv_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, num_workers=0,
        shuffle=True, collate_fn=collate_fn)

    # cpu
    start_time = time.time()
    fake_training_loop(dataloader, folder = './cpu')
    print(f'on CPU {time.time()-start_time}')

    # cuda
    start_time = time.time()
    fake_training_loop(dataloader, 'cuda', folder = './cuda')
    print(f'on GPU {time.time()-start_time}')
   
    # cuda + batch_size
    dataloader = DataLoader(dataset, batch_size=64, num_workers=0,
        shuffle=True, collate_fn=collate_fn)
    start_time = time.time()
    fake_training_loop(dataloader, 'cuda', folder = './cuda_batch')
    print(f'on GPU batch size 64 {time.time()-start_time}')

    # cuda + batch_size + numworkers
    dataloader = DataLoader(dataset, batch_size=64, num_workers=16,
        shuffle=True, collate_fn=collate_fn)
    start_time = time.time()
    fake_training_loop(dataloader, 'cuda', folder = './cuda_batch_numworkers')
    print(f'on GPU batch size 64 {time.time()-start_time}')

    print('done!')