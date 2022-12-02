import torch
import torch.nn.functional as F
from torchvision import transforms

import numpy as np

from PIL import Image, ImageDraw

from random import uniform, randint, choices, shuffle

import os

from tqdm import tqdm

from config import class_imgid_list, class_name_list, imgid_filename_dict, background_img_path_list, imgid_bbox_dict

import albumentations as A


# albumentation_augmentation 함수에서 쓰이는 A.Compose 변수
albumentation_transform = A.Compose([
    A.SafeRotate(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
    A.AdvancedBlur(p=0.3),
    A.RandomFog(p=0.3),
], bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"]))


# 조류 이미지를 자르기 전에 augmentation을 하는 함수.
def albumentation_augmentation(
        img,
        segmentation,
        bbox
    ):

    augmentation_result = albuemtnation_transform(img=img, mask=segmentation, bboxes=[bbox])

    return augmentation_result["image"], augmentation_result["mask"], augmentation_result["bboxes"][0]


# 조류 이미지를 Bounding Box만큼 자르는 함수.
def crop_img(
        bird_img,
        segmentation,
        bounding_box: tuple
    ):

    # (xmin, ymin, w, h)를 (xmin, ymin, xmax, ymax)로 바꾸는 과정
    xmin, ymin, w, h = map(int, bounding_box)
    xmax, ymax = xmin+w, ymin+h
    
    # (xmin, ymin, xmax, ymax)의 영역만을 반환.
    return bird_img[:, ymin:ymax, xmin:xmax], segmentation[:, ymin:ymax, xmin:xmax]


# 랜덤하게 이미지를 resize하는 함수. (비율 고정)
def random_resize(
        bird_img,
        segmentation,
        min_scale_factor=0.25,
        max_scale_factor=1.75,
        max_size=256,
        mode="bilinear"
    ):

    _, bird_H, bird_W = bird_img.shape # 조류 이미지의 세로 길이, 가로 길이

    min_sc = min_scale_factor
    max_sc = max_scale_factor

    if max(bird_H, bird_W) * max_scale_factor > max_size:
        max_sc = max_size / max(bird_H, bird_W)

    # min_scale_factor = min_size / min(bird_H, bird_W) # 최소 scale factor
    # max_scale_factor = max_size / max(bird_H, bird_W) # 최대 scale factor

    scale_factor = uniform(min_sc, max_sc) # 균등 분포에서 랜덤한 scale factor 추출.

    resized_bird_img = F.interpolate(bird_img.unsqueeze(0), scale_factor=scale_factor, mode=mode).squeeze(0) # 조류 이미지 resize
    resized_segmentation = F.interpolate(segmentation.unsqueeze(0), scale_factor=scale_factor, mode=mode).squeeze(0) # segmentation 이미지 resize

    return resized_bird_img, resized_segmentation # resize한 이미지 반환.


# 조류를 삽입할 랜덤한 위치를 반환하는 함수.
def get_random_position(
        bird_img,
        segmentation,
        background_img
    ):

    _, bird_H, bird_W = bird_img.shape # 조류 이미지의 세로 길이, 가로 길이
    _, background_H, background_W = background_img.shape # 배경 이미지의 세로 길이, 가로 길이

    # 만약 조류 이미지가 배경 이미지보다 크다면, 에러 발생시킴.
    if bird_H > background_H or bird_W > background_W:
        raise Exception(f"the size of bird_img({bird_H}, {bird_W}) must be smaller than the size of background_img({background_H, background_W})")

    max_xmin = background_W - bird_W
    max_ymin = background_H - bird_H

    xmin = randint(0, max_xmin)
    ymin = randint(0, max_ymin)

    return [xmin, ymin, xmin+bird_W, ymin+bird_H]


# 조류를 특정 위치에 삽입하는 함수.
def paste_img(bird_img, segmentation, background_img, position):
    xmin, ymin, xmax, ymax = position

    # segmentation 이미지를 이용하여 해당하는 조류 이미지의 크기에 해당하는 부분만 결합함.
    img = bird_img * segmentation + background_img[:, ymin:ymax, xmin:xmax] * (1 - segmentation)

    # pasted_img와 background_img가 서로 같은 메모리를 공유하지 않기 위함.
    pasted_img = background_img.clone() # 이렇게 하지 않으면 삽입한 조류 이미지가 계속 남아 누적됨.

    # 결합한 이미지를 삽입함.
    pasted_img[:, ymin:ymax, xmin:xmax] = img

    return pasted_img


# 조류 이미지를 증강하는 클래스
class BirdAugmentation:

    def __init__(
        self,
        min_scale_factor=0.25, # 조류 이미지를 resize할 때 사용할 변수
        max_scale_factor=1.75, # 조류 이미지를 resize할 때 사용할 변수
        max_size=256, # 조류 이미지를 resize할 때 사용할 변수
        resize_mode="bilinear", # 조류 이미지를 resize할 때 사용할 변수
        birds_n: list = [0, 1, 2], # 한 이미지에 들어갈 수 있는 조류의 마릿수 리스트
        birds_n_p: list = [0.01, 0.65, 0.34], # 위 변수에 각각 대응되는 확률 리스트
        ioa_threshold: float = 0.5 # 조류 이미지를 얼마나 겹치게 하는 것을 허용할지 결정하는 변수.
    ):

        self.min_scale_factor = min_scale_factor
        self.max_scale_factor = max_scale_factor
        self.max_size = max_size

        self.resize_mode = resize_mode # resize에 사용될 resize_mode

        self.birds_n = birds_n # 하나의 이미지에 들어가는 조류의 개수 리스트
        self.birds_n_p = birds_n_p # 조류의 개수에 대한 확률 값

        self.ioa_threshold = ioa_threshold # intersection over area


    def augmentation_img(
        self,
        birds_p: list,
        bird_img_list: list,
        segmentation_list: list,
        bounding_box_list: list,
        classes: list,
        background_imgs: list,
        save_file_dir: str,
        save_file_path: str,
        is_train: bool,
    ):

        random_background = background_imgs[randint(0, len(background_imgs)-1)]
        [random_birds_n] = choices(self.birds_n, weights=self.birds_n_p, k=1)

        bounding_boxes = []

        pasted_img = random_background.clone()

        for _ in range(random_birds_n):
            
            [random_idx] = choices(range(len(bird_img_list)), weights=birds_p, k=1)
            random_bird = bird_img_list[random_idx].type(torch.float32)
            random_segmentation = segmentation_list[random_idx].type(torch.float32)
            random_bbox = bounding_box_list[random_idx]

            try:
                random_bird, random_segmentation, random_bbox = albumentation_augmentation(
                    random_bird,
                    random_segmentation,
                    random_bbox
                )
            except: pass

            random_class = classes[random_idx]

            croped_bird_img, croped_segmentation = crop_img(random_bird, random_segmentation, random_bbox)

            resized_bird, resized_segmentation = random_resize(
                bird_img=croped_bird_img,
                segmentation=croped_segmentation,
                min_scale_factor=self.min_scale_factor,
                max_scale_factor=self.max_scale_factor,
                max_size=self.max_size,
                mode=self.resize_mode
            )
            
            random_position = get_random_position(resized_bird, resized_segmentation, random_background)

            its_ok = True

            for _, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax in bounding_boxes:
                rand_xmin, rand_ymin, rand_xmax, rand_ymax = random_position

                intersection = (min(bbox_xmax, rand_xmax) - max(bbox_xmin, rand_xmin)) * (min(bbox_ymax, rand_ymax) - max(bbox_ymin, rand_ymin))
                area = (bbox_xmax - bbox_xmin) * (bbox_ymax - bbox_ymin)

                ioa = intersection / area
                if ioa >= self.ioa_threshold:
                    its_ok = False
                    break

            if not its_ok: continue

            pasted_img = paste_img(resized_bird, resized_segmentation, pasted_img, random_position)

            bounding_boxes.append([random_class, *random_position])

        pasted_img = transforms.ToPILImage()(pasted_img)

        img_path = os.path.join(save_file_dir, "train" if is_train else "test", "images", f"{save_file_path}.jpg")
        pasted_img.save(img_path)

        for i in range(len(bounding_boxes)):
            x = (bounding_boxes[i][1] + bounding_boxes[i][3]) / 2
            y = (bounding_boxes[i][2] + bounding_boxes[i][4]) / 2
            w = (bounding_boxes[i][3] - bounding_boxes[i][1])
            h = (bounding_boxes[i][4] - bounding_boxes[i][2])

            x /= random_background.size(2)
            y /= random_background.size(1)
            w /= random_background.size(2)
            h /= random_background.size(1)

            bounding_boxes[i] = f"{bounding_boxes[i][0]} {x} {y} {w} {h}"

        label_path = os.path.join(save_file_dir, "train" if is_train else "test", "labels", f"{save_file_path}.txt")
        with open(label_path, "w") as f:
            f.write("\n".join(bounding_boxes) + "\n")


def draw_bbox(img, bbox, format="xywh"):
    image = transforms.ToPILImage()(img)
    _, H, W = img.shape

    draw = ImageDraw.Draw(image)

    if format == "xywh":
        for x, y, w, h in bbox:
            xmin = (x - (w/2)) * W
            ymin = (y - (h/2)) * H
            xmax = (x + (w/2)) * W
            ymax = (y + (h/2)) * H

            draw.rectangle((xmin, ymin, xmax, ymax), outline=(255, 0, 0), width=3)
        
    elif format == "minmax":
        for xmin, ymin, xmax, ymax in bbox:
            draw.rectangle((xmin*W, ymin*H, xmax*W, ymax*H), outline=(255, 0, 0), width=3)
    
    return image


if __name__ == '__main__':

    # ----------------------------------------- augmentation images ---------------------------------------

    # 사용할 class들.
    class_list = [
        [29, 30, 107, 108], 
        [31, 33],
        [113, 123, 127, 129, 130],
        [136, 137, 138],
        [188, 189, 190, 191, 192],
    ]

    number_of_img_class = [0 for _ in range(len(class_list))]

    for i, classes in enumerate(class_list):
        for c in classes:
            number_of_img_class[i] += len(class_imgid_list[c])

    class_convert_dict = {}
    class_name_dict = {
        0: "Crow", # 까마귀
        1: "Cuckoo", # 뻐꾸기
        2: "Sparrow", # 참새
        3: "Swallow", # 제비
        4: "Woodpecker", # 딱따구리
    }

    for i, class_ in enumerate(class_list):
        for c in class_:
            class_convert_dict[c] = i

    min_scale_factor = 0.125
    max_scale_factor = 1.75
    max_size = 256
    resize_mode = "bilinear" # resize에 사용할 resize_mode

    background_img_size = (256, 256) # 배경 이미지의 크기 (HxW)

    # 배경 이미지 tensor.
    background_img_dataset = torch.zeros((len(background_img_path_list), 3, *background_img_size)).type(torch.float16)

    print("load background_imgs")
    # 배경 이미지 경로 리스트를 모두 돌아서 배경 이미지 tensor를 채움.
    for i, background_path in tqdm(enumerate(background_img_path_list)):
        background_img_dataset[i] = transforms.ToTensor()(Image.open(background_path).convert("RGB").resize(background_img_size)).type(torch.float16)
    

    # 아래의 변수들은 같은 인덱스에 서로 매칭됨.

    # 사용할 조류 이미지들
    bird_img_list = []
    # 사용할 segmentation 이미지 파일들의 리스트
    segmentation_img_list = []
    # 사용할 조류 이미지의 bounding_box(xmin, ymin, xmax, ymax) 리스트
    bounding_box_list = []
    # 사용할 조류 이미지의 class 리스트
    class_dataset = []
    # 조류 idx의 확률
    birds_p = []

    print("load imgs")
    # 사용할 class만큼 반복
    for i, class_ in tqdm(enumerate(class_list)):
        for c in class_:
            # 사용할 class에 해당하는 이미지의 id를 모두 반복.
            for imgid in class_imgid_list[c]:
                # 해당하는 imgid의 file_name
                file_name = imgid_filename_dict[imgid]

                # 조류 이미지 경로
                bird_path = os.path.join(
                    "CUB200", "images", class_name_list[c], file_name
                )

                bird_img_list.append(
                    transforms.ToTensor()(Image.open(bird_path).convert("RGB")).type(torch.float16)
                )

                # 세그멘테이션 이미지 경로
                segmentation_path = os.path.join(
                    "CUB200", "segmentations", class_name_list[c], file_name.replace("jpg", "png")
                )

                segmentation_img_list.append(
                    transforms.ToTensor()(Image.open(segmentation_path).convert("RGB")).type(torch.float16)
                )

                # imgid에 해당하는 bounding_box를 추가함.
                bounding_box_list.append(
                    imgid_bbox_dict[imgid]
                )

                # class_에 c를 해준 다음에 추가함.
                class_dataset.append(
                    class_convert_dict[c]
                )

                birds_p.append(
                    1/(len(class_list)*number_of_img_class[i])
                )
                
    is_train = [0 if i < 0.1*len(bird_img_list) else 1 for i in range(len(bird_img_list))]
    shuffle(is_train)

    trainset_size = 40000
    testset_size = 4000

    bird_augmentation = BirdAugmentation(
        min_scale_factor=min_scale_factor,
        max_scale_factor=max_scale_factor,
        max_size=max_size,
        resize_mode=resize_mode,
        birds_n=list(range(8)),
        birds_n_p=[0.125] * 8
    )

    print("trainset")
    for i in tqdm(range(trainset_size)):
        bird_augmentation.augmentation_img(
            birds_p=[birds_p[i] for i in range(len(birds_p)) if is_train[i]],
            bird_img_list=[bird_img_list[i] for i in range(len(bird_img_list)) if is_train[i]],
            segmentation_list=[segmentation_img_list[i] for i in range(len(segmentation_img_list)) if is_train[i]],
            bounding_box_list=[bounding_box_list[i] for i in range(len(bounding_box_list)) if is_train[i]],
            classes=[class_dataset[i] for i in range(len(class_dataset)) if is_train[i]],
            background_imgs=background_img_dataset,
            save_file_dir="./dataset",
            save_file_path=str(i),
            is_train=True
        )

    print("testset")
    for i in tqdm(range(testset_size)):
        bird_augmentation.augmentation_img(
            birds_p=[birds_p[i] for i in range(len(birds_p)) if not is_train[i]],
            bird_img_list=[bird_img_list[i] for i in range(len(bird_img_list)) if not is_train[i]],
            segmentation_list=[segmentation_img_list[i] for i in range(len(segmentation_img_list)) if not is_train[i]],
            bounding_box_list=[bounding_box_list[i] for i in range(len(bounding_box_list)) if not is_train[i]],
            classes=[class_dataset[i] for i in range(len(class_dataset)) if not is_train[i]],
            background_imgs=background_img_dataset,
            save_file_dir="./dataset",
            save_file_path=str(i),
            is_train=False
        )
    

    # # # ----------------------------------------- add CUB images ---------------------------------------

    
    albumentation_transform = A.Compose([
        A.SafeRotate(p=0.5),
        A.CropAndPad(percent=(-0.2,0.4), p=1),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.8),
        A.AdvancedBlur(p=0.3),
        A.RandomFog(p=0.3),
        A.Resize(height=256,width=256,p=1)
    ], bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"]))


    print("add CUB images (train)")
    cnt = 0
    for i, (bird_img, bbox, c) in tqdm(enumerate(zip(bird_img_list, bounding_box_list, class_dataset))):
        if is_train[i]:
            for _ in range(30):
                try:
                    augmented_result = albumentation_transform(
                        image=torch.permute(bird_img.type(torch.float32), (1, 2, 0)).numpy(),
                        bboxes=[bbox],
                        class_labels=[c]
                    )
                    xmin, ymin, w, h = augmented_result["bboxes"][0]
                    
                    x = xmin + w/2
                    y = ymin + h/2

                    x /= 256
                    y /= 256
                    w /= 256
                    h /= 256

                    img = transforms.ToPILImage()(torch.from_numpy(augmented_result["image"]).permute(2, 0, 1))

                    img.save(f"./dataset/train/images/{trainset_size+cnt}.jpg")
                    with open(f"./dataset/train/labels/{trainset_size+cnt}.txt", "w") as f:
                        f.write(f"{c} {x} {y} {w} {h}\n")
                
                except:
                    pass

                cnt += 1

    
    print("add CUB images (test)")
    cnt = 0
    for i, (bird_img, bbox, c) in tqdm(enumerate(zip(bird_img_list, bounding_box_list, class_dataset))):
        if not is_train[i]:
            for _ in range(15):
                try:
                    augmented_result = albumentation_transform(
                        image=torch.permute(bird_img.type(torch.float32), (1, 2, 0)).numpy(),
                        bboxes=[bbox],
                        class_labels=[c]
                    )
                    xmin, ymin, w, h = augmented_result["bboxes"][0]
                    
                    x = xmin + w/2
                    y = ymin + h/2

                    x /= 256
                    y /= 256
                    w /= 256
                    h /= 256

                    img = transforms.ToPILImage()(torch.from_numpy(augmented_result["image"]).permute(2, 0, 1))

                    img.save(f"./dataset/test/images/{testset_size+cnt}.jpg")
                    with open(f"./dataset/test/labels/{testset_size+cnt}.txt", "w") as f:
                        f.write(f"{c} {x} {y} {w} {h}\n")
                
                except:
                    pass

                cnt += 1

    
    # ------------------------------------- make CUB200_test_images ---------------------------------------
    
    print("make CUB200_test_images")
    for i, (bird_img, bbox, c) in tqdm(enumerate(zip(bird_img_list, bounding_box_list, class_dataset))):
        if not is_train[i]:
            xmin, ymin, w, h = bbox
            
            x = xmin + w/2
            y = ymin + h/2

            _, H, W = bird_img.shape

            x /= W
            y /= H
            w /= W
            h /= H

            img = transforms.ToPILImage()(bird_img)

            img.save(f"./CUB200_test_images/images/{i}.jpg")
            with open(f"./CUB200_test_images/labels/{i}.txt", "w") as f:
                f.write(f"{c} {x} {y} {w} {h}\n")
    
    # # ----------------------------------------- check train image ---------------------------------------
    # img = transforms.ToTensor()(Image.open("./dataset/train/images/10001.jpg"))
    # bbox = []
    
    # with open("./dataset/train/labels/10001.txt", "r") as f:
    #     for line in f.readlines():
    #         if line != "\n":
    #             c, x, y, w, h = map(float, line.split())
    #             bbox.append([x, y, w, h])
    
    # img = draw_bbox(img, bbox)

    # img.show()