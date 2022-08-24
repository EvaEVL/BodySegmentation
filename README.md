# BodySegmentation
Репозиторий хранит реализацию на pytorch U-net модели, в основе которой Resnet-18.

# Насущное
У меня появился вопрос, как работает сегментация изображений и как устроена архитектура модели U-net. Чтобы ответить на свои вопросы, я принялся реализовывать модель для сегментации человеческого тела.
# Данные

Для обучения модели был использован [кегловский датасет](https://www.kaggle.com/datasets/tapakah68/segmentation-full-body-mads-dataset).

# Подготовка Dataset & DataLoader

Для увеличения исходного набора данных была добавлена аугментация: горизонтальный поворот, поворот на угол и искажение цветовой палитры изображения.

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

    # Random horizontal flipping
    if random.random() >= 0.5:

        img1 = TF.hflip(img1)
        img2 = TF.hflip(img2)
    
    # Random rotation
    if random.random() >= 0.5:
        angle = random.randint(0, 91)
        img1 = transforms.functional.rotate(img1, angle)
        img2 = transforms.functional.rotate(img2, angle)
        
    # Randomly change the brightness, contrast, saturation and hue
    if random.random() >= 0.5:
        colorJitter = transforms.ColorJitter(brightness = (0.1, 0.8), contrast = (0.1, 0.8), saturation = (0.1, 0.8), hue = (-0.2, 0.2))
        img1 = colorJitter(img1)
    

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
# Изображения из датасета

```python

def display_image_grid(images_filenames, images_directory, masks_directory, predict_masks=False, num_of_images=BATCH_SIZE):
  cols = 3 if predict_masks else 2
  rows = num_of_images
  figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 24))
  random.shuffle(images_filenames)

  for i, image_filename in enumerate(images_filenames):
    if i >= rows:
      break

    image = read_image(os.path.join(images_directory, image_filename))
    mask = read_image(os.path.join(masks_directory, image_filename), mode = io.ImageReadMode(1)).permute(1, 2, 0)

    ax[i, 0].imshow(image.permute(1, 2, 0))
    ax[i, 1].imshow(mask.squeeze(), 'gray')

    ax[i, 0].set_title("Image")
    ax[i, 1].set_title("Ground truth mask")

    ax[i, 0].set_axis_off()
    ax[i, 1].set_axis_off()
  
    if predict_masks:
      ima, _ = val_transform(image, torch.rand((1,1,224,224)))
      predicted_mask = predict(model, ima).detach().cpu().permute(1,2,0).numpy() > 0.5
      ax[i, 2].imshow(predicted_mask.squeeze(), 'gray')
      ax[i, 2].set_title("Predicted mask")
      ax[i, 2].set_axis_off()

  plt.tight_layout()
  plt.show()

display_image_grid(images_filenames=list(sorted(os.listdir(image_path))), images_directory=image_path, masks_directory=mask_path, num_of_images=4)
```
![image](https://user-images.githubusercontent.com/24653067/186390044-650e4788-4f36-42cc-a79f-820c85089e02.png)
На изображении сверху видно, каким выглядит исходное изображение и маска (соответственно, слева и справа).

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
print_image(train_features[ra], train_labels[ra])
```

![image](https://user-images.githubusercontent.com/24653067/186390496-88012d6a-4a2f-4896-a8c5-05541bccb97d.png)

На картинке сверху Dataloader передает предобработанные изображения, откуда и появляется зашумление левой картинки. Слева - преобработанное исходное изображение, а справа - маска.

# Создание U-net модели

```python
def convrelu(in_channels, out_channels, kernel=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )
    
class UnetModelResNet(nn.Module):
    #Unet model based on pretrained Restnet 18
    def __init__(self, n_class=1):
        super(UnetModelResNet, self).__init__()
        
        self.resnet_model = torchvision.models.resnet18(pretrained=True)
        self.resnet_layers = list(self.resnet_model.children())
        
        for param in self.resnet_model.parameters():
            param.requires_grad = False
        
        self.block0 = nn.Sequential(*self.resnet_layers[:3])
        self.block0_1x1 = convrelu(64, 64)
        self.block1 = nn.Sequential(*self.resnet_layers[3:5])
        self.block1_1x1 = convrelu(64, 64)
        self.block2 = nn.Sequential(*self.resnet_layers[5])
        self.block2_1x1 = convrelu(128, 128)
        self.block3 = nn.Sequential(*self.resnet_layers[6])
        self.block3_1x1 = convrelu(256, 256)
        self.block4 = nn.Sequential(*self.resnet_layers[7])
        self.block4_1x1 = convrelu(512, 512)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.relu = torch.nn.ReLU() 
        
        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)
        
        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)
        
        self.conv_last = nn.Conv2d(64, n_class, 1, padding=0)
        
    def forward(self, x):
        x_original = self.conv_original_size0(x)
        x_original = self.conv_original_size1(x_original)
        
        
        block0 = self.block0(x)
        block1 = self.block1(block0)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        
        block4 = self.block4_1x1(block4)
        x = self.upsample(block4)
        block3 = self.block3_1x1(block3)
        x = torch.cat([x, block3], dim=1)
        x = self.conv_up3(x)
        
        x = self.upsample(x)
        block2 = self.block2_1x1(block2)
        x = torch.cat([x, block2], dim=1)
        x = self.conv_up2(x)
        
        x = self.upsample(x)
        block1 = self.block1_1x1(block1)
        x = torch.cat([x, block1], dim=1)
        x = self.conv_up1(x)
        
        x = self.upsample(x)
        block0 = self.block0_1x1(block0)
        x = torch.cat([x, block0], dim=1)
        x = self.conv_up0(x)
        
        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out
        
```

# Описание модели

```python

model = UnetModelResNet()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: ', device)
model = model.to(device)

summary(model, (3, 224, 224))
```

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 224, 224]           1,792
              ReLU-2         [-1, 64, 224, 224]               0
            Conv2d-3         [-1, 64, 224, 224]          36,928
              ReLU-4         [-1, 64, 224, 224]               0
            Conv2d-5         [-1, 64, 112, 112]           9,408
            Conv2d-6         [-1, 64, 112, 112]           9,408
       BatchNorm2d-7         [-1, 64, 112, 112]             128
       BatchNorm2d-8         [-1, 64, 112, 112]             128
              ReLU-9         [-1, 64, 112, 112]               0
             ReLU-10         [-1, 64, 112, 112]               0
        MaxPool2d-11           [-1, 64, 56, 56]               0
        MaxPool2d-12           [-1, 64, 56, 56]               0
           Conv2d-13           [-1, 64, 56, 56]          36,864
           Conv2d-14           [-1, 64, 56, 56]          36,864
      BatchNorm2d-15           [-1, 64, 56, 56]             128
      BatchNorm2d-16           [-1, 64, 56, 56]             128
             ReLU-17           [-1, 64, 56, 56]               0
             ReLU-18           [-1, 64, 56, 56]               0
           Conv2d-19           [-1, 64, 56, 56]          36,864
           Conv2d-20           [-1, 64, 56, 56]          36,864
      BatchNorm2d-21           [-1, 64, 56, 56]             128
      BatchNorm2d-22           [-1, 64, 56, 56]             128
             ReLU-23           [-1, 64, 56, 56]               0
             ReLU-24           [-1, 64, 56, 56]               0
       BasicBlock-25           [-1, 64, 56, 56]               0
       BasicBlock-26           [-1, 64, 56, 56]               0
           Conv2d-27           [-1, 64, 56, 56]          36,864
           Conv2d-28           [-1, 64, 56, 56]          36,864
      BatchNorm2d-29           [-1, 64, 56, 56]             128
      BatchNorm2d-30           [-1, 64, 56, 56]             128
             ReLU-31           [-1, 64, 56, 56]               0
             ReLU-32           [-1, 64, 56, 56]               0
           Conv2d-33           [-1, 64, 56, 56]          36,864
           Conv2d-34           [-1, 64, 56, 56]          36,864
      BatchNorm2d-35           [-1, 64, 56, 56]             128
      BatchNorm2d-36           [-1, 64, 56, 56]             128
             ReLU-37           [-1, 64, 56, 56]               0
             ReLU-38           [-1, 64, 56, 56]               0
       BasicBlock-39           [-1, 64, 56, 56]               0
       BasicBlock-40           [-1, 64, 56, 56]               0
           Conv2d-41          [-1, 128, 28, 28]          73,728
           Conv2d-42          [-1, 128, 28, 28]          73,728
      BatchNorm2d-43          [-1, 128, 28, 28]             256
      BatchNorm2d-44          [-1, 128, 28, 28]             256
             ReLU-45          [-1, 128, 28, 28]               0
             ReLU-46          [-1, 128, 28, 28]               0
           Conv2d-47          [-1, 128, 28, 28]         147,456
           Conv2d-48          [-1, 128, 28, 28]         147,456
      BatchNorm2d-49          [-1, 128, 28, 28]             256
      BatchNorm2d-50          [-1, 128, 28, 28]             256
           Conv2d-51          [-1, 128, 28, 28]           8,192
           Conv2d-52          [-1, 128, 28, 28]           8,192
      BatchNorm2d-53          [-1, 128, 28, 28]             256
      BatchNorm2d-54          [-1, 128, 28, 28]             256
             ReLU-55          [-1, 128, 28, 28]               0
             ReLU-56          [-1, 128, 28, 28]               0
       BasicBlock-57          [-1, 128, 28, 28]               0
       BasicBlock-58          [-1, 128, 28, 28]               0
           Conv2d-59          [-1, 128, 28, 28]         147,456
           Conv2d-60          [-1, 128, 28, 28]         147,456
      BatchNorm2d-61          [-1, 128, 28, 28]             256
      BatchNorm2d-62          [-1, 128, 28, 28]             256
             ReLU-63          [-1, 128, 28, 28]               0
             ReLU-64          [-1, 128, 28, 28]               0
           Conv2d-65          [-1, 128, 28, 28]         147,456
           Conv2d-66          [-1, 128, 28, 28]         147,456
      BatchNorm2d-67          [-1, 128, 28, 28]             256
      BatchNorm2d-68          [-1, 128, 28, 28]             256
             ReLU-69          [-1, 128, 28, 28]               0
             ReLU-70          [-1, 128, 28, 28]               0
       BasicBlock-71          [-1, 128, 28, 28]               0
       BasicBlock-72          [-1, 128, 28, 28]               0
           Conv2d-73          [-1, 256, 14, 14]         294,912
           Conv2d-74          [-1, 256, 14, 14]         294,912
      BatchNorm2d-75          [-1, 256, 14, 14]             512
      BatchNorm2d-76          [-1, 256, 14, 14]             512
             ReLU-77          [-1, 256, 14, 14]               0
             ReLU-78          [-1, 256, 14, 14]               0
           Conv2d-79          [-1, 256, 14, 14]         589,824
           Conv2d-80          [-1, 256, 14, 14]         589,824
      BatchNorm2d-81          [-1, 256, 14, 14]             512
      BatchNorm2d-82          [-1, 256, 14, 14]             512
           Conv2d-83          [-1, 256, 14, 14]          32,768
           Conv2d-84          [-1, 256, 14, 14]          32,768
      BatchNorm2d-85          [-1, 256, 14, 14]             512
      BatchNorm2d-86          [-1, 256, 14, 14]             512
             ReLU-87          [-1, 256, 14, 14]               0
             ReLU-88          [-1, 256, 14, 14]               0
       BasicBlock-89          [-1, 256, 14, 14]               0
       BasicBlock-90          [-1, 256, 14, 14]               0
           Conv2d-91          [-1, 256, 14, 14]         589,824
           Conv2d-92          [-1, 256, 14, 14]         589,824
      BatchNorm2d-93          [-1, 256, 14, 14]             512
      BatchNorm2d-94          [-1, 256, 14, 14]             512
             ReLU-95          [-1, 256, 14, 14]               0
             ReLU-96          [-1, 256, 14, 14]               0
           Conv2d-97          [-1, 256, 14, 14]         589,824
           Conv2d-98          [-1, 256, 14, 14]         589,824
      BatchNorm2d-99          [-1, 256, 14, 14]             512
     BatchNorm2d-100          [-1, 256, 14, 14]             512
            ReLU-101          [-1, 256, 14, 14]               0
            ReLU-102          [-1, 256, 14, 14]               0
      BasicBlock-103          [-1, 256, 14, 14]               0
      BasicBlock-104          [-1, 256, 14, 14]               0
          Conv2d-105            [-1, 512, 7, 7]       1,179,648
          Conv2d-106            [-1, 512, 7, 7]       1,179,648
     BatchNorm2d-107            [-1, 512, 7, 7]           1,024
     BatchNorm2d-108            [-1, 512, 7, 7]           1,024
            ReLU-109            [-1, 512, 7, 7]               0
            ReLU-110            [-1, 512, 7, 7]               0
          Conv2d-111            [-1, 512, 7, 7]       2,359,296
          Conv2d-112            [-1, 512, 7, 7]       2,359,296
     BatchNorm2d-113            [-1, 512, 7, 7]           1,024
     BatchNorm2d-114            [-1, 512, 7, 7]           1,024
          Conv2d-115            [-1, 512, 7, 7]         131,072
          Conv2d-116            [-1, 512, 7, 7]         131,072
     BatchNorm2d-117            [-1, 512, 7, 7]           1,024
     BatchNorm2d-118            [-1, 512, 7, 7]           1,024
            ReLU-119            [-1, 512, 7, 7]               0
            ReLU-120            [-1, 512, 7, 7]               0
      BasicBlock-121            [-1, 512, 7, 7]               0
      BasicBlock-122            [-1, 512, 7, 7]               0
          Conv2d-123            [-1, 512, 7, 7]       2,359,296
          Conv2d-124            [-1, 512, 7, 7]       2,359,296
     BatchNorm2d-125            [-1, 512, 7, 7]           1,024
     BatchNorm2d-126            [-1, 512, 7, 7]           1,024
            ReLU-127            [-1, 512, 7, 7]               0
            ReLU-128            [-1, 512, 7, 7]               0
          Conv2d-129            [-1, 512, 7, 7]       2,359,296
          Conv2d-130            [-1, 512, 7, 7]       2,359,296
     BatchNorm2d-131            [-1, 512, 7, 7]           1,024
     BatchNorm2d-132            [-1, 512, 7, 7]           1,024
            ReLU-133            [-1, 512, 7, 7]               0
            ReLU-134            [-1, 512, 7, 7]               0
      BasicBlock-135            [-1, 512, 7, 7]               0
      BasicBlock-136            [-1, 512, 7, 7]               0
          Conv2d-137            [-1, 512, 7, 7]         262,656
            ReLU-138            [-1, 512, 7, 7]               0
        Upsample-139          [-1, 512, 14, 14]               0
          Conv2d-140          [-1, 256, 14, 14]          65,792
            ReLU-141          [-1, 256, 14, 14]               0
          Conv2d-142          [-1, 512, 14, 14]       3,539,456
            ReLU-143          [-1, 512, 14, 14]               0
        Upsample-144          [-1, 512, 28, 28]               0
          Conv2d-145          [-1, 128, 28, 28]          16,512
            ReLU-146          [-1, 128, 28, 28]               0
          Conv2d-147          [-1, 256, 28, 28]       1,474,816
            ReLU-148          [-1, 256, 28, 28]               0
        Upsample-149          [-1, 256, 56, 56]               0
          Conv2d-150           [-1, 64, 56, 56]           4,160
            ReLU-151           [-1, 64, 56, 56]               0
          Conv2d-152          [-1, 256, 56, 56]         737,536
            ReLU-153          [-1, 256, 56, 56]               0
        Upsample-154        [-1, 256, 112, 112]               0
          Conv2d-155         [-1, 64, 112, 112]           4,160
            ReLU-156         [-1, 64, 112, 112]               0
          Conv2d-157        [-1, 128, 112, 112]         368,768
            ReLU-158        [-1, 128, 112, 112]               0
        Upsample-159        [-1, 128, 224, 224]               0
          Conv2d-160         [-1, 64, 224, 224]         110,656
            ReLU-161         [-1, 64, 224, 224]               0
          Conv2d-162          [-1, 1, 224, 224]              65
================================================================
Total params: 28,976,321
Trainable params: 6,623,297
Non-trainable params: 22,353,024
----------------------------------------------------------------
Input size (MB): 0.57
Forward/backward pass size (MB): 415.73
Params size (MB): 110.54
Estimated Total Size (MB): 526.84
----------------------------------------------------------------
```

# Создание обучающего цикла и метрик оценивания
Для оценивания качества работы модели были выбраны две матрики:
1) [IoU(intersection over union)](https://en.wikipedia.org/wiki/Jaccard_index) - число от 0 до 1, показывающее, насколько у двух объектов (эталонного (ground true) и текущего) совпадает внутренность.
2)[Dice coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient) - величина, показывающая схожесть двух множеств.




``` python 

def ioU_metric(predicted, ground_truth, threshold = 0.5, smooth=1):
    predicted = (predicted > threshold)
    intersection = (predicted * threshold).sum(dim=(1,2,3))
    union = predicted.sum(dim=(1,2,3)) + ground_truth.sum(dim=(1,2,3)) - intersection
    iou =  ((intersection + smooth) / (union + smooth)).mean()
    return iou.item()

def dice_coefficient_metric(predicted, ground_truth, threshold = 0.5, smooth=1):
    predicted = (predicted > threshold)
    intersection = (predicted * threshold).sum(dim=(1,2,3))
    union = predicted.sum(dim=(1,2,3)) + ground_truth.sum(dim=(1,2,3)) - intersection
    f1 = ((2.0 * intersection + smooth) / (smooth + union)).mean()
    return f1.item()
```


Обучеющий цикл

``` python 

def train_model(model, optimizer, loss, scheduler, num_epoch=1, batch_size=8):

    dataloaders = {
        'train': train_dl,
        'val': valid_dl
    }

    best_loss = 1e9
    best_model = copy.deepcopy(model.state_dict())
    
    history = {'train' : [],
               'val' : []}
    
    iou_metric = { 
            'train' : [],
            'val' : []}
    
    f1metric = { 
            'train' : [],
            'val' : []}
    
    epoch_samples = 0
    m = torch.nn.Sigmoid()
    
    for epoch in range(num_epoch):
        print('Epoch {}/{} \n'.format(epoch + 1, num_epoch))
        print('-' * 10)
        time_start = time.time()
        
        
        
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            running_iou = 0.0
            running_f1 = 0.0
            
            
            for inputs, labels in dataloaders[phase]:#tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(float).to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    #forward
                    outputs = model(inputs)
                    loss_value = loss(m(outputs).to(float), labels)
                    
                    #backward
                    if phase == 'train':
                        loss_value.backward()
                        optimizer.step()
                
                running_loss += loss_value.data.cpu().numpy()
                running_iou += ioU_metric(outputs, labels)
                running_f1 += f1_metric(outputs, labels)
                
                
            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_iou = running_iou / len(dataloaders[phase])
            epoch_f1 = running_f1 / len(dataloaders[phase])
            
            
            print('{} Loss: {:.8f} '.format(phase, epoch_loss))
            history[phase].append(epoch_loss)
            iou_metric[phase].append(epoch_iou)
            f1metric[phase].append(epoch_f1)
            
            
            if phase == 'val' and best_loss > loss_value.item():
                best_loss = loss_value.item()
                best_model = copy.deepcopy(model.state_dict())
                torch.save(best_model, os.getcwd() + '/weights_best')
            
        time_check = time_start - time.time()
        print('{:.0f}m {:.0f}s'.format(time_check // 60, time_check % 60), flush=True)
        
    print('best loss : {}'.format(best_loss))
    model.load_state_dict(best_model)
        
    return model, history, iou_metric, f1metric
    
```

# Обучение
В нашей задачи стоит вопрос соотнесения пикселя к классу тела человека, поэтому в качетсве функции потерь использовалась [бинарная кроссэнтропия](https://en.wikipedia.org/wiki/Cross_entropy). В качестве был взят adam, затухание обучающего коэффициента в 10 раз происходит каждые 5 эпох.

``` python 

pretrained = True
weights_path = "/content/drive/MyDrive/weights_best_colab"
if pretrained:
  model.load_state_dict(torch.load(weights_path, map_location=device))  
else:
  model, history, iou_metric, f1metric = train_model(model = model, optimizer = optimizer, loss = loss, scheduler = scheduler, num_epoch=num_epoch);
```

# Визуализация



``` python 

def plot_graphs(num_epoch, history, act):
    x = np.arange(num_epoch)
    plt.plot(x, history['train'],label = 'train', )
    plt.plot(x, history['val'], label = 'validation', )
    plt.title(f'{act} while training/validation')
    plt.xlabel('Epoch')
    plt.ylabel(act)
    plt.legend()
    plt.show()
    return

```

``` python
plot_graphs(num_epoch, history, 'BCELoss')

 ```
 ![image](https://user-images.githubusercontent.com/24653067/185366391-dc95b8ff-4f3d-40c5-8eb7-eba00a0aad49.png)

 ```  python
 plot_graphs(num_epoch,  iou_metric, 'IOU')
  ``` 
 ![image](https://user-images.githubusercontent.com/24653067/185366554-6b2efece-e692-4b7f-81d4-0049758106c7.png)

  
  ``` python
  plot_graphs(num_epoch,  f1metric, 'F1')
  ```
  ![image](https://user-images.githubusercontent.com/24653067/185366660-26e4740b-7a86-420c-a78b-c5f78f8ddb94.png)


# Использование обученное модели

 ``` python
 
 def predict(model, inputs):
    inputs = torch.unsqueeze(inputs, dim=0)
    pred = model(inputs.to(device))
    return torch.squeeze(pred, 0)
    
import urllib.request
image_url = "https://cdn3.whatculture.com/images/2021/08/5e21602b99b1ba27-600x338.jpeg"
image_name = 'picture.jpg'

urllib.request.urlretrieve(image_url, image_name)
 
image = read_image('./' + image_name)

img_reshaped = image.permute(1, 2, 0)

ima, _ = transform(image, torch.rand((1,1,224,224)))
pre = predict(model, ima).detach().cpu().permute(1,2,0).numpy()

print_image(img_reshaped.numpy(), pre > 0.5)
``` 
![image](https://user-images.githubusercontent.com/24653067/185369091-877d340a-d656-4fa3-9416-a3fa634bd35e.png)

# Заключение

Поведение модели на незнакомых данных показывает не лучшие результаты; в качестве улучшения обобщяющей способности модели следует добавить аугментацию.

