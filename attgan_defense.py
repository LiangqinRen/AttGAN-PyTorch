import argparse
import json
import os
from os.path import join
import random

import torch
import torch.utils.data as data
import torchvision.utils as vutils
import torch.nn as nn

from attgan import AttGAN
from helpers import Progressbar
from data import check_attribute_conflict
from utils import find_model
from torchvision.utils import save_image


class AttGANDefense(nn.Module):
    # sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

    def __init__(self, logger, args):
        super(AttGANDefense, self).__init__()

        self.project_path = os.path.dirname(os.path.abspath(__file__))
        self.dataset_path = ()

        self.logger = logger
        self.args = args

        # project related args
        self.args.experiment = "128_shortcut1_inject1_none"
        self.args.load_epoch = "latest"

        os.chdir(os.path.dirname(os.path.realpath(__file__)))
        with open(join("output", self.args.experiment, "setting.txt"), "r") as f:
            args_ = json.load(f, object_hook=lambda d: argparse.Namespace(**d))

        self.args = argparse.Namespace(**vars(args), **vars(args_))
        self.target = AttGAN(self.args)
        self.target.load(
            find_model(
                join("output", self.args.experiment_name, "checkpoint"),
                self.args.load_epoch,
            )
        )

    def _get_random_img_path(self, batch_size: int = 1) -> list[str]:
        imgs = os.listdir(self.args.data_path)
        selected_imgs = random.sample(imgs, batch_size)
        selected_imgs = [f"{self.args.data_path}/{img}" for img in selected_imgs]

        return selected_imgs

    def _get_dataloader(self) -> data.DataLoader:
        if self.args.data == "CelebA":
            from data import CelebA

            test_dataset = CelebA(
                self.args.data_path,
                self.args.attr_path,
                self.args.img_size,
                "test",
                self.args.attrs,
            )
        if self.args.data == "CelebA-HQ":
            from data import CelebA_HQ

            test_dataset = CelebA_HQ(
                self.args.data_path,
                self.args.attr_path,
                self.args.image_list_path,
                self.args.img_size,
                "test",
                self.args.attrs,
            )

        test_dataloader = data.DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=self.args.num_workers,
            shuffle=True,
            drop_last=False,
        )

        return test_dataloader

    def _denormalize(self, results: torch.tensor) -> torch.tensor:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        for i in range(len(results)):
            for j in range(len(mean)):
                results[i][j] = results[i][j] * std[j] + mean[j]

        return results

    def void(self, args):
        test_count = 5
        save_path = f"../../log/{args.ID}/{args.project}_void.png"

        self.target.eval()
        test_dataloader = self._get_dataloader()
        results = []
        for i, (img, att) in enumerate(test_dataloader):
            if i >= test_count:
                break

            img = img.cuda()
            att = att.cuda()

            att_all = [att]
            for i in range(self.args.n_attrs):
                tmp = att.clone()
                tmp[:, i] = 1 - tmp[:, i]
                tmp = check_attribute_conflict(tmp, self.args.attrs[i], self.args.attrs)
                att_all.append(tmp)

            with torch.no_grad():
                edited_imgs = [img]
                for i, att in enumerate(att_all):
                    att_ = (att * 2 - 1) * self.args.thres_int
                    if i > 0:
                        att_[..., i - 1] = (
                            att_[..., i - 1] * self.args.test_int / self.args.thres_int
                        )
                    edited_img = self.target.G(img, att_)
                    edited_imgs.append(edited_img)
                edited_imgs = torch.cat(edited_imgs, dim=0)
                results.append(edited_imgs)
        result = torch.cat(results, dim=0)
        save_image(self._denormalize(result), save_path, nrow=self.args.n_attrs + 1 + 1)
