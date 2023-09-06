import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import CelebA, Caltech256, CIFAR100, ImageFolder
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, Subset
from src.models.cnn import SimpleCNNAE
from PIL import Image
import os
from PVQ import PVQ

def get_model(args):
    if args.model == "cnn":
        args.encoder_depth = 2
        args.decoder_depth = 2
        args.encoder_inner_channels = 256
        args.decoder_inner_channels = 256

        return SimpleCNNAE(args)

    if args.model == "cnn_small_decoder":
        args.encoder_depth = 2
        args.decoder_depth = 2
        args.encoder_inner_channels = 64
        args.decoder_inner_channels = 16

        return SimpleCNNAE(args)




def get_dataset(args):
    transform = []
    transform.append(transforms.Resize((args.image_dim, args.image_dim), antialias=True))
    transform.append(transforms.ToTensor())

    if args.augmentation:
        transform.append(transforms.RandAugment())

    transform.append(transforms.ConvertImageDtype(torch.float32))


    args.image_channels = 3

    if args.dataset == "celeba":
        train = CelebA(root=args.data_root, download=True, transform=transforms.Compose(transform))
        test = CelebA(root=args.data_root, split="valid", download=True, transform=transforms.Compose(transform))
        return train, test

    if args.dataset == "cifar100":
        train = CIFAR100(root=args.data_root, download=True, transform=transforms.Compose(transform))
        test = CIFAR100(root=args.data_root, train=False, download=True, transform=transforms.Compose(transform))
        return train, test

    if args.dataset == "unsplash":
        root = os.path.join(args.data_root, "unsplash")
        ds = ImageFolder(root=root, transform=transforms.Compose(transform))
        train, test = torch.utils.data.random_split(ds, (0.8, 0.2))
        return train, test



def get_dataloader(dataset, args):
    return DataLoader(dataset, batch_size=args.batch_size, num_workers=4)


def get_optimizer_scheduler(args, model):
    if args.optim == "adam":
        optim = torch.optim.Adam(model.parameters(), args.lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=args.step_size, gamma=args.gamma)
    return optim, scheduler


class Averager:
    def __init__(self):
        self.sum = 0
        self.n = 0

    def add(self, x):
        self.sum += x
        self.n += 1

    def item(self):
        if self.n == 0:
            return 0
        return self.sum / self.n


def visualize_results(model, epoch, args):
    model.eval()
    with torch.no_grad():
        exemplars = [torch.tensor(np.asarray(Image.open(f"./data/{i}.jpg"))) for i in range(3)]
        exemplars = (torch.stack(exemplars).permute(0, 3, 1, 2).float() / 255.).to(args.device)

        exemplars = transforms.Resize((256, 256))(exemplars)

        latent = model.forward_encoder(exemplars)

        latent = latent.flatten(1)

        latent_size = latent[0].numel()
        encoder_params = sum([param.numel() for param in model.encoder.parameters()])
        decoder_params = sum([param.numel() for param in model.decoder.parameters()])

        pvq = PVQ(4, 7, "./data/pvq")
        for i, lat in enumerate(latent):
            encoded = pvq.encode(lat.cpu().detach().numpy())
            decoded = pvq.decode(encoded)
            latent[i] = torch.tensor(decoded)

        latent = latent.unflatten(-1, (args.latent_channel, 64, 64))
        preds = model.forward_decoder(latent)

        grid_tensor = torch.concatenate([exemplars, preds], 0)

        grid = make_grid(grid_tensor, 3).permute(1, 2, 0).cpu().detach().numpy()

        ax = plt.gca()
        ax.set_title(f"Epoch {epoch}\nencoder params: {encoder_params}, decoder params: {decoder_params}\nlatent: {latent_size}")
        plt.imsave(os.path.join(args.model_save_path, f"result_epoch_{epoch}.png"), grid)

    model.train()
