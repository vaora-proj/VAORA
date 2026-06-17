#!/usr/bin/env python3
from __future__ import annotations

import glob
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision


# PHYRE DQN checkpoints are trained with creator.constants.NUM_COLORS (=7).
NUM_COLORS = 7
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ActionNetwork(nn.Module):
    def __init__(self, action_size, output_size, hidden_size=256, num_layers=1):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(action_size, hidden_size)])
        for _ in range(1, num_layers):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, tensor):
        for layer in self.layers:
            tensor = nn.functional.relu(layer(tensor), inplace=True)
        return self.output(tensor)


class FilmActionNetwork(nn.Module):
    def __init__(self, action_size, output_size, **kwargs):
        super().__init__()
        self.net = ActionNetwork(action_size, output_size * 2, **kwargs)

    def forward(self, actions, image):
        beta, gamma = torch.chunk(
            self.net(actions).unsqueeze(-1).unsqueeze(-1), chunks=2, dim=1
        )
        return image * beta + gamma


class SimpleNetWithAction(nn.Module):
    def __init__(self, action_size, action_network_kwargs=None):
        super().__init__()
        action_network_kwargs = action_network_kwargs or {}
        self.stem = nn.Sequential(
            nn.Conv2d(NUM_COLORS, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 64, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.action_net = ActionNetwork(action_size, 128, **action_network_kwargs)

    @property
    def device(self):
        return "cuda" if next(self.parameters()).is_cuda else "cpu"

    def preprocess(self, observations):
        image = self._image_colors_to_onehot(observations.to(dtype=torch.long, device=self.device))
        return dict(features=self.stem(image).squeeze(-1).squeeze(-1))

    def forward(self, observations, actions, preprocessed=None):
        if preprocessed is None:
            preprocessed = self.preprocess(observations)
        return self._forward(actions, **preprocessed)

    def _forward(self, actions, features):
        actions = self.action_net(actions.to(features.device))
        return (actions * features).sum(-1) / (actions.shape[-1] ** 0.5)

    def _image_colors_to_onehot(self, indices):
        onehot = torch.nn.functional.embedding(
            indices, torch.eye(NUM_COLORS, device=indices.device)
        )
        return onehot.permute(0, 3, 1, 2).contiguous()


class ResNet18FilmAction(nn.Module):
    def __init__(self, action_size, action_layers=1, action_hidden_size=256, fusion_place="last"):
        super().__init__()
        net = torchvision.models.resnet18(pretrained=False)
        conv1 = nn.Conv2d(NUM_COLORS, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.register_buffer("embed_weights", torch.eye(NUM_COLORS))
        self.stem = nn.Sequential(conv1, net.bn1, net.relu, net.maxpool)
        self.stages = nn.ModuleList([net.layer1, net.layer2, net.layer3, net.layer4])

        def build_film(output_size):
            return FilmActionNetwork(
                action_size,
                output_size,
                hidden_size=action_hidden_size,
                num_layers=action_layers,
            )

        self.last_network = None
        if fusion_place == "all":
            self.action_networks = nn.ModuleList([build_film(size) for size in (64, 64, 128, 256)])
        elif fusion_place == "last":
            self._action_network = build_film(256)
            self.action_networks = [None, None, None, self._action_network]
        elif fusion_place == "first":
            self._action_network = build_film(64)
            self.action_networks = [self._action_network, None, None, None]
        elif fusion_place == "last_single":
            self.last_network = build_film(512)
            self.action_networks = [None, None, None, None]
        elif fusion_place == "none":
            self.action_networks = [None, None, None, None]
        else:
            raise ValueError(f"Unknown fusion place: {fusion_place}")
        self.reason = nn.Linear(512, 1)

    @property
    def device(self):
        return "cuda" if next(self.parameters()).is_cuda else "cpu"

    def preprocess(self, observations):
        image = self._image_colors_to_onehot(observations)
        features = self.stem(image)
        for stage, act_layer in zip(self.stages, self.action_networks):
            if act_layer is not None:
                break
            features = stage(features)
        else:
            features = nn.functional.adaptive_max_pool2d(features, 1)
        return dict(features=features)

    def forward(self, observations, actions, preprocessed=None):
        if preprocessed is None:
            preprocessed = self.preprocess(observations)
        return self._forward(actions, **preprocessed)

    def _forward(self, actions, features):
        actions = actions.to(features.device)
        skip_compute = True
        for stage, film_layer in zip(self.stages, self.action_networks):
            if film_layer is not None:
                skip_compute = False
                features = film_layer(actions, features)
            if skip_compute:
                continue
            features = stage(features)
        if not skip_compute:
            features = nn.functional.adaptive_max_pool2d(features, 1)
        if self.last_network is not None:
            features = self.last_network(actions, features)
        features = features.flatten(1)
        if features.shape[0] == 1 and actions.shape[0] != 1:
            features = features.expand(actions.shape[0], -1)
        return self.reason(features).squeeze(-1)

    def _image_colors_to_onehot(self, indices):
        onehot = torch.nn.functional.embedding(
            indices.to(dtype=torch.long, device=self.embed_weights.device),
            self.embed_weights,
        )
        return onehot.permute(0, 3, 1, 2).contiguous()


def build_model(network_type: str, **kwargs):
    if network_type == "resnet18":
        return ResNet18FilmAction(
            kwargs["action_space_dim"],
            fusion_place=kwargs["fusion_place"],
            action_hidden_size=kwargs["action_hidden_size"],
            action_layers=kwargs["action_layers"],
        )
    if network_type == "simple":
        return SimpleNetWithAction(kwargs["action_space_dim"])
    raise ValueError(f"Unknown network type: {network_type}")


def _latest_checkpoint(folder: str) -> Optional[str]:
    checkpoints = sorted(glob.glob(os.path.join(folder, "ckpt.*")))
    return checkpoints[-1] if checkpoints else None


def load_model_from_phyre_ckpt(path_or_folder: str):
    ckpt_path = path_or_folder
    if os.path.isdir(path_or_folder):
        ckpt_path = _latest_checkpoint(path_or_folder)
        if ckpt_path is None:
            raise FileNotFoundError(f"No ckpt.* found in folder: {path_or_folder}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    load_kwargs = dict(map_location=DEVICE)
    try:
        checkpoint = torch.load(ckpt_path, weights_only=False, **load_kwargs)
    except TypeError:
        # Older torch versions do not support weights_only kwarg.
        checkpoint = torch.load(ckpt_path, **load_kwargs)
    model = build_model(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model"])
    model.to(DEVICE)
    model.eval()
    return model


def eval_actions(model, actions: np.ndarray, batch_size: int, observation: np.ndarray) -> np.ndarray:
    scores = []
    with torch.no_grad():
        preprocessed = model.preprocess(torch.LongTensor(observation).unsqueeze(0))
        for batch_start in range(0, len(actions), batch_size):
            batch_end = min(len(actions), batch_start + batch_size)
            batch_actions = torch.FloatTensor(actions[batch_start:batch_end]).to(DEVICE)
            batch_scores = model(None, batch_actions, preprocessed=preprocessed)
            scores.append(batch_scores.detach().cpu().numpy())
    return np.concatenate(scores, axis=0)
