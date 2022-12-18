from config import class_imgid_list, class_name_list, imgid_filename_dict

import os

import torch

from tqdm import tqdm

from PIL import Image

from torchvision import transforms

from random import shuffle

import albumentations as A



if __name__ == '__main__':
    root_dir = os.path.join("/", "home", "kdhsimplepro", "kdhsimplepro", "AI", "BirdAugmentation")
    bird_img_dir = os.path.join(root_dir, "CUB200", "images")
    mask_img_dir = os.path.join(root_dir, "CUB200", "segmentations")

    train_ratio = 0.8

    save_bird_img_dir = os.path.join(root_dir, "BirdSegmentation", "dataset", "bird_img")
    save_mask_img_dir = os.path.join(root_dir, "BirdSegmentation", "dataset", "mask_img")


    # 사용할 클래스 리스트
    class_list = [29, 30, 107, 108, 31, 33, 113, 123, 127, 129, 130, 136, 137, 138, 188, 189, 190, 191, 192]

    augmentation_n = 5

    dataset_list = []

    albumentation_transform = A.Compose([
        A.SafeRotate(p=0.3),
        A.CropAndPad(percent=(-0.2,0.4), p=1),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.8),
        A.AdvancedBlur(p=0.2),
        A.RandomFog(p=0.2),
        A.Resize(height=256,width=256,p=1)
    ])


    for c in tqdm(class_list):

        classname = class_name_list[c]

        for imgid in class_imgid_list[c]:
            filename = imgid_filename_dict[imgid]

            bird_path = os.path.join(bird_img_dir, classname, filename)
            mask_path = os.path.join(mask_img_dir, classname, filename.replace("jpg", "png"))

            bird_img = transforms.ToTensor()(Image.open(bird_path).convert("RGB"))
            mask_img = transforms.ToTensor()(Image.open(mask_path).convert("L"))

            # (channel_n, H, W) tensor -> (H, W, channel_n) np array
            bird_img = bird_img.permute(1, 2, 0).numpy()
            mask_img = mask_img.permute(1, 2, 0).numpy()

            for _ in range(augmentation_n):
                augmentation_result = albumentation_transform(image=bird_img, mask=mask_img)

                dataset_list.append(
                    # (H, W, channel_n) np array -> (channel_n, H, W) tensor
                    (
                        torch.from_numpy(augmentation_result["image"]).permute(2, 0, 1),
                        torch.from_numpy(augmentation_result["mask"]).permute(2, 0, 1)
                    )
                )


    is_train = [1 if i < (len(dataset_list) * train_ratio) else 0 for i in range(len(dataset_list))]
    shuffle(is_train)

    for i, (bird_img, mask_img) in tqdm(enumerate(dataset_list)):
        save_bird_img_path = os.path.join(save_bird_img_dir, "train" if is_train[i] else "valid", f"{i}.jpg")
        save_mask_img_path = os.path.join(save_mask_img_dir, "train" if is_train[i] else "valid", f"{i}.jpg")

        transforms.ToPILImage()(bird_img).save(save_bird_img_path)
        transforms.ToPILImage()(mask_img).save(save_mask_img_path)
