import torch

from torch.optim import Adam

from torch.nn import BCELoss

from torch.utils.data import DataLoader, TensorDataset, random_split

from torchvision import transforms
from torchvision.utils import make_grid

import pickle

from argparse import ArgumentParser

from math import sqrt

import wandb

from tqdm import tqdm

from unet import UNet
from deeplab_v3plus import DeepLabV3plus

from glob import glob

import os

from PIL import Image

import matplotlib.pyplot as plt


class SegmentationTrainer:

    def __init__(
        self,
        device,
        model,
        trainloader,
        validloader,
        learning_rate=0.002,
        could_wandb=False
    ):

        self.device = device

        self.model = model.to(device)
        self.optim = Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = BCELoss()

        self.best_loss = 1e10
        self.best_state_dict = self.model.state_dict()
        self.model_state_dict = self.model.state_dict()

        self.trainloader = trainloader
        self.validloader = validloader

        self.train_loss = []
        self.valid_loss = []

        self.test_images = []

        self.test_x, self.test_y = next(iter(self.validloader))
        self.test_x = self.test_x.to(device)
        self.test_y = self.test_y.to(device)

        self.could_wandb = could_wandb

    
    @torch.no_grad()
    def test(self):
        self.model.eval()

        nrow = int(sqrt(self.test_x.size(0)))

        pred = self.model(self.test_x)

        x_grid = make_grid(self.test_x.detach().cpu(), nrow=nrow, normalize=True)
        y_grid = make_grid(self.test_y.detach().cpu(), nrow=nrow, normalize=True)
        pred_grid = make_grid(pred.detach().cpu(), nrow=nrow, normalize=True)

        grid = torch.cat((x_grid, y_grid, pred_grid), dim=2)
        image = transforms.ToPILImage()(grid)

        return image

    
    @torch.no_grad()
    def valid_epoch(self):
        self.model.eval()

        avg_loss = 0

        for x, y in tqdm(self.validloader):
            x = x.to(self.device)
            y = y.to(self.device)

            pred = self.model(x)

            loss = self.criterion(pred, y)

            avg_loss += loss.item()

            if self.could_wandb:
                wandb.log({"valid_loss": loss.item()})

            self.valid_loss.append(loss.item())


        avg_loss /= len(self.validloader)

        if avg_loss <= self.best_loss:
            self.best_loss = avg_loss
            self.best_state_dict = self.model.state_dict()
        
        self.model_state_dict = self.model.state_dict()

        return avg_loss

    
    def train_epoch(self):
        self.model.train()

        avg_loss = 0

        for x, y in tqdm(self.trainloader):
            x = x.to(self.device)
            y = y.to(self.device)

            pred = self.model(x)

            loss = self.criterion(pred, y)

            avg_loss += loss.item()

            if self.could_wandb: 
                wandb.log({"train_loss": loss.item()})

            self.train_loss.append(loss.item())


            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        avg_loss /= len(self.trainloader)

        return avg_loss

    

    def load_trainer(self, path):

        with open(path, "rb") as f:
            trainer_data = pickle.load(f)

        self.model.load_state_dict(trainer_data["model_state_dict"])
        self.optim = trainer_data["optim"]
        
        self.best_loss = trainer_data["best_loss"]
        self.best_state_dict = trainer_data["best_state_dict"]

        self.train_loss = trainer_data["train_loss"]
        self.valid_loss = trainer_data["valid_loss"]

        self.test_images = trainer_data["test_images"]
        self.test_x = trainer_data["test_x"]
        self.test_y = trainer_data["test_y"]
        

    
    def save_traier(self, path):

        trainer_data = {
            "model_state_dict": self.model_state_dict,
            "optim": self.optim,
            "best_loss": self.best_loss,
            "best_state_dict": self.best_state_dict,
            "train_loss": self.train_loss,
            "valid_loss": self.valid_loss,
            "test_images": self.test_images,
            "test_x": self.test_x,
            "test_y": self.test_y
        }

        with open(path, "wb") as f:
            pickle.dump(trainer_data, f, pickle.HIGHEST_PROTOCOL)


    def run(self, epochs:tuple=(0, 20), save_path: str="./trainer_data.pickle"):

        for epoch in range(*epochs):

            print(f"\n\nEPOCH: [{epoch+1}/{epochs[1]}]\n")
            
            print("TRAIN")
            train_loss = self.train_epoch()

            print("VALID")
            valid_loss = self.valid_epoch()

            print(f"train_loss: {train_loss}, valid_loss: {valid_loss}")


            test_img = self.test()
            self.test_images.append(test_img)

            if self.could_wandb:
                wandb.log({"test_images": wandb.Image(test_img)})

            self.save_traier(path=save_path)




if __name__ == '__main__':

    parser = ArgumentParser(description="BirdSegmentation hyper parameter")

    parser.add_argument("--learning_rate", '-l', default=0.001, type=float, help="learning rate of Model")

    parser.add_argument("--batch_size", default=16, type=int, help="batch size that used while model train and evalauate")

    parser.add_argument("--epochs", "-e", default=(0, 20), type=str, help="epochs that type is tuple(range), for example 0-10")

    parser.add_argument("--project_dir", default="/home/kdhsimplepro/kdhsimplepro/AI/BirdAugmentation/", type=str, help="BirdAugmentation project directory path")

    parser.add_argument("--could_wandb", default=False, type=bool, help="It must be True, if you want to log history in wandb")

    parser.add_argument("--wandb_project_name", default="BirdSegmentation", type=str, help="wandb_project_name")

    parser.add_argument("--wandb_run_name", default="run0", type=str, help="wandb_run_name")

    parser.add_argument("--model_type", default="unet", type=str, help="model that will be used to train type (unet or deeplabv3+)")

    parser.add_argument("--trainer_name", default="trainer_data", type=str, help="trainer name that will be used to save trainer data")

    args = parser.parse_args()

    args.epochs = tuple(map(int, args.epochs.split("-")))

    if args.could_wandb:
        wandb.init(project=args.wandb_project_name, reinit=True)
        wandb.run.name = args.wandb_run_name
        wandb.config.update(args)
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet() if args.model_type == "unet" else DeepLabV3plus()

    train_bird_img_paths = glob(os.path.join(args.project_dir, "BirdSegmentation", "dataset", "bird_img", "train", "*.jpg"))
    valid_bird_img_paths = glob(os.path.join(args.project_dir, "BirdSegmentation", "dataset", "bird_img", "valid", "*.jpg"))
    train_mask_img_paths = glob(os.path.join(args.project_dir, "BirdSegmentation", "dataset", "mask_img", "train", "*.jpg"))
    valid_mask_img_paths = glob(os.path.join(args.project_dir, "BirdSegmentation", "dataset", "mask_img", "valid", "*.jpg"))

    train_x, train_y = torch.zeros((len(train_bird_img_paths), 3, 256, 256)), torch.zeros((len(train_mask_img_paths), 1, 256, 256))
    valid_x, valid_y = torch.zeros((len(valid_bird_img_paths), 3, 256, 256)), torch.zeros((len(valid_mask_img_paths), 1, 256, 256))

    for i in tqdm(range(len(train_bird_img_paths))):
        train_x[i] = transforms.ToTensor()(Image.open(train_bird_img_paths[i]))
        train_y[i] = transforms.ToTensor()(Image.open(train_mask_img_paths[i]).convert("L"))

    for i in tqdm(range(len(valid_bird_img_paths))):
        valid_x[i] = transforms.ToTensor()(Image.open(valid_bird_img_paths[i]))
        valid_y[i] = transforms.ToTensor()(Image.open(valid_mask_img_paths[i]).convert("L"))

    trainset = TensorDataset(train_x, train_y)
    validset = TensorDataset(valid_x, valid_y)

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
    validloader = DataLoader(validset, batch_size=args.batch_size*2, shuffle=True)


    trainer = SegmentationTrainer(
        device=device,
        model=model,
        trainloader=trainloader,
        validloader=validloader,
        learning_rate=args.learning_rate,
        could_wandb=args.could_wandb
    )

    trainer.run(epochs=args.epochs, save_path=os.path.join(args.project_dir, "BirdSegmentation", f"{args.trainer_name}.pickle"))

    # plt.plot(range(len(trainer.train_loss)), trainer.train_loss, label="train_loss")
    # plt.plot(range(len(trainer.valid_loss)), trainer.valid_loss, label="valid_loss")
    # plt.show()