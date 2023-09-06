import os

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
import tqdm
from src.metrics import cw_n01
from utils import *


class Trainer:
    def __init__(self, args):
        self.args = args
        self.ensure_path_exists()

        self.dataset_train, self.dataset_test = get_dataset(args)
        self.model = get_model(args).to(self.args.device)
        self.optim, self.scheduler = get_optimizer_scheduler(args, self.model)

        self.dataloader_train = get_dataloader(self.dataset_train, args)
        self.dataloader_test = get_dataloader(self.dataset_test, args)

        if args.content_loss == "l2":
            self.criterion = F.mse_loss
        elif args.content_loss == "l1":
            self.criterion = F.l1_loss

        visualize_results(self.model, 0, args)

    def train(self):
        best_test_loss = 1e20
        for epoch in range(self.args.epochs):
            print("Starting epoch", epoch)
            content_loss_avg = Averager()
            cw_loss_avg = Averager()
            for x, y in tqdm.tqdm(self.dataloader_train):
                x = x.to(self.args.device)
                latent, pred = self.model(x)
                content_loss = self.criterion(pred, x)
                cw_loss = self.get_cw_loss(latent)

                loss = content_loss + self.args.cw_coeff * cw_loss

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                content_loss_avg.add(content_loss.item())
                cw_loss_avg.add(cw_loss.item())

            print("Epoch", epoch, "end")
            print("content loss:", content_loss_avg.item())
            print("cw loss:", cw_loss_avg.item())

            if epoch % self.args.saving_interval == self.args.saving_interval - 1:
                self.save_checkpoint(epoch)
                visualize_results(self.model, epoch, self.args)


            test_loss = self.test()

            print("test loss:", test_loss)
            print("best test loss:", best_test_loss)

            if test_loss < best_test_loss:
                print("Better model found! saving...")
                self.save_checkpoint("best")
                best_test_loss = test_loss

    def get_cw_loss(self, latent):
        if not self.args.cw:
            return torch.zeros(1).to(self.args.device)

        with torch.no_grad():
            gamma = torch.square(1.059 * latent.std()) * torch.pow(latent.size(0), torch.tensor(-0.4))

        return cw_n01(latent.view(latent.shape[0], -1), gamma)

    def test(self):
        loss_avg = Averager()
        for x, y in self.dataloader_test:
            x = x.to(self.args.device)
            latent, pred = self.model(x)
            content_loss = self.criterion(pred, x)
            cw_loss = self.get_cw_loss(latent)

            loss = content_loss + self.args.cw_coeff * cw_loss
            loss_avg.add(loss.item())

        return loss_avg.item()

    def save_checkpoint(self, epoch):
        model_save_path = os.path.join(self.args.model_save_path, f"checkpoint_{epoch}_{self.args.epochs}.pt")
        torch.save({
            "model": self.model.state_dict(),
            "optim": self.optim.state_dict()
        }, model_save_path)

    def load_checkpoint(self, epoch):
        model_save_path = os.path.join(self.args.model_save_path, f"checkpoint_{epoch}_{self.args.epochs}.pt")
        dicts = torch.load(model_save_path)
        self.model.load_state_dict(dicts["model"])
        self.optim.load_state_dict(dicts["optim"])

    def ensure_path_exists(self):
        if not os.path.exists(self.args.model_save_path):
            os.makedirs(self.args.model_save_path)