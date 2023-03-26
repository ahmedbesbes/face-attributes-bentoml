from torchvision import transforms

train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5063, 0.4258, 0.3832], std=[0.2644, 0.2436, 0.2397]
        ),
    ]
)

valid_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.5063, 0.4258, 0.3832], std=[0.2644, 0.2436, 0.2397]
        ),
    ]
)
