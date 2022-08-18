# BodySegmentation
Репозиторий хранит реализацию на pytorch U-net модели, в основе которой находится Resnet-18.

# Данные

В качестве задачи была взята сегментация человеческого тела, а для обучения модели был использован [кегловский датасет](https://www.kaggle.com/datasets/tapakah68/segmentation-full-body-mads-dataset).

# Подготовка Dataset & DataLoader

```python
IMAGE_SIZE = (224, 224)


class SegData(Dataset):
    def __init__(self, root_path, transform=None):
        super().__init__()
        
        self.transform = transform
        self.img_files = glob.glob(os.path.join(root_path,'images','*.png'))
        self.mask_files = []
        for img_path in self.img_files:
             self.mask_files.append(os.path.join(root_path,'masks',os.path.basename(img_path)))

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        image_path = self.img_files[index]
        mask_path  = self.mask_files[index]
        image = read_image(image_path)
        mask = read_image(mask_path, mode = io.ImageReadMode(1))#read mode 1 - Gray
        
        if self.transform:
            image, mask = self.transform(image, mask)
            
        return image, mask

def transform(img1, img2):
    
    params = transforms.RandomResizedCrop.get_params(img1, scale=(0.5, 1.0), ratio=(0.75, 1.33))
    img1 = TF.resized_crop(img1, *params, size=IMAGE_SIZE)
    img2 = TF.resized_crop(img2, *params, size=IMAGE_SIZE)

    
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img1 = train_transforms(img1)
    
    img2 = img2.to(torch.float) / 255
    
    return img1, img2
    
data = SegData(data_root, transform)
BATCH_SIZE = 8

train_data, valid_data = torch.utils.data.random_split(dataset = data, lengths= [1000, len(data) - 1000])

train_dl = DataLoader(
        train_data,
        BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
    )


valid_dl = DataLoader(
        valid_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        pin_memory=True,
    )
```
# Проверка загруженных изображений Dataloader'а

```python

def print_image(img, mask):
    plt.figure(figsize=(12, 5))
    plt.subplot(121)
    plt.imshow(TF.to_pil_image(img))
    plt.subplot(122)
    plt.imshow(mask.squeeze(),'gray')
    plt.show()
    return
    
train_features, train_labels = next(iter(train_dl))
ra = random.randint(0, 7)
print(train_features.shape, train_labels.shape)
```
![image](https://user-images.githubusercontent.com/24653067/185358659-b527065c-1954-445f-b8c7-5f4f276af8e6.png)
