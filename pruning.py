import random
import os
import re
from collections import OrderedDict, defaultdict
import copy
import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from torch import nn
from torch.optim import *
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader
from torchprofile import profile_macs
from torchvision.datasets import *
from torchvision.transforms import *
from tqdm.auto import tqdm
from typing import Union, List

def get_model_macs(model, inputs) -> int:
    return profile_macs(model, inputs)


def get_sparsity(tensor: torch.Tensor) -> float:
    """
    calculate the sparsity of the given tensor
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    """
    return 1 - float(tensor.count_nonzero()) / tensor.numel()


def get_model_sparsity(model: nn.Module) -> float:
    """
    calculate the sparsity of the given model
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    """
    num_nonzeros, num_elements = 0, 0
    for param in model.parameters():
        num_nonzeros += param.count_nonzero()
        num_elements += param.numel()
    return 1 - float(num_nonzeros) / num_elements

def get_num_parameters(model: nn.Module, count_nonzero_only=False) -> int:
    """
    calculate the total number of parameters of model
    :param count_nonzero_only: only count nonzero weights
    """
    num_counted_elements = 0
    for param in model.parameters():
        if count_nonzero_only:
            num_counted_elements += param.count_nonzero()
        else:
            num_counted_elements += param.numel()
    return num_counted_elements


def get_model_size(model: nn.Module, data_width=32, count_nonzero_only=False) -> int:
    """
    calculate the model size in bits
    :param data_width: #bits per element
    :param count_nonzero_only: only count nonzero weights
    """
    return get_num_parameters(model, count_nonzero_only) * data_width

Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB
GiB = 1024 * MiB

def conv_block_1(in_dim,out_dim, activation,stride=1):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim, kernel_size=1, stride=stride),
        nn.BatchNorm2d(out_dim),
        activation,
    )
    return model

def conv_block_3(in_dim,out_dim, activation, stride=1):
    model = nn.Sequential(
        nn.Conv2d(in_dim,out_dim, kernel_size=3, stride=stride, padding=1),
        nn.BatchNorm2d(out_dim),
        activation,
    )
    return model

class BottleNeck(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim, activation, down=False):
        super(BottleNeck, self).__init__()
        self.down = down

        # 특성지도의 크기가 감소하는 경우
        if self.down:
            self.layer = nn.Sequential(
                conv_block_1(in_dim, mid_dim, activation, stride=2),
                conv_block_3(mid_dim, mid_dim, activation, stride=1),
                conv_block_1(mid_dim, out_dim, activation, stride=1),
            )

            # 특성지도 크기 + 채널을 맞춰주는 부분
            self.downsample = nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=2)

        # 특성지도의 크기가 그대로인 경우
        else:
            self.layer = nn.Sequential(
                conv_block_1(in_dim, mid_dim, activation, stride=1),
                conv_block_3(mid_dim, mid_dim, activation, stride=1),
                conv_block_1(mid_dim, out_dim, activation, stride=1),
            )

        # 채널을 맞춰주는 부분
        self.dim_equalizer = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(self, x):
        if self.down:
            downsample = self.downsample(x)
            out = self.layer(x)
            out = out + downsample
        else:
            out = self.layer(x)
            if x.size() is not out.size():
                x = self.dim_equalizer(x)
            out = out + x
        return out

class ResNet(nn.Module):
    def __init__(self, base_dim = 64, num_classes=10):
        super(ResNet, self).__init__()
        self.activation = nn.ReLU()
        self.layer_1 = nn.Sequential(
            nn.Conv2d(3, base_dim, 7, 2, 3),
            nn.ReLU(),
            nn.MaxPool2d(3, 2, 1),
        )
        self.layer_2 = nn.Sequential(
            BottleNeck(base_dim, base_dim, base_dim * 4, self.activation),
            BottleNeck(base_dim * 4, base_dim, base_dim * 4, self.activation),
            BottleNeck(base_dim * 4, base_dim, base_dim * 4, self.activation, down=True),
        )
        self.layer_3 = nn.Sequential(
            BottleNeck(base_dim * 4, base_dim * 2, base_dim * 8, self.activation),
            BottleNeck(base_dim * 8, base_dim * 2, base_dim * 8, self.activation),
            BottleNeck(base_dim * 8, base_dim * 2, base_dim * 8, self.activation),
            BottleNeck(base_dim * 8, base_dim * 2, base_dim * 8, self.activation, down=True),
        )
        self.layer_4 = nn.Sequential(
            BottleNeck(base_dim * 8, base_dim * 4, base_dim * 16, self.activation),
            BottleNeck(base_dim * 16, base_dim * 4, base_dim * 16, self.activation),
            BottleNeck(base_dim * 16, base_dim * 4, base_dim * 16, self.activation),
            BottleNeck(base_dim * 16, base_dim * 4, base_dim * 16, self.activation),
            BottleNeck(base_dim * 16, base_dim * 4, base_dim * 16, self.activation),
            BottleNeck(base_dim * 16, base_dim * 4, base_dim * 16, self.activation, down=True),
        )
        self.layer_5 = nn.Sequential(
            BottleNeck(base_dim * 16, base_dim * 8, base_dim * 32, self.activation),
            BottleNeck(base_dim * 32, base_dim * 8, base_dim * 32, self.activation),
            BottleNeck(base_dim * 32, base_dim * 8, base_dim * 32, self.activation),
        )
        self.avgpool = nn.AvgPool2d(1, 1)
        self.fc_layer = nn.Linear(base_dim * 32, num_classes)

    def forward(self, x):
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = self.layer_5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc_layer(out)
        return out

model = ResNet().cuda()

@torch.inference_mode()
def evaluate(model: nn.Module, dataloader: DataLoader,criterion: nn.Module, verbose=True, ) -> float:
    model.eval()

    num_samples = 0
    num_correct = 0
    epoch_loss = 0.0
    total_samples = 0

    for inputs, targets in tqdm(dataloader, desc="eval", leave=False, disable=not verbose):
        # Move the data from CPU to GPU
        inputs = inputs.cuda()
        targets = targets.cuda()

        # Inference
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Convert logits to class indices
        outputs = outputs.argmax(dim=1)

        batch_size = inputs.size(0)
        epoch_loss += loss.item() * batch_size
        total_samples += batch_size

        # Update metrics
        num_samples += targets.size(0)
        num_correct += (outputs == targets).sum()
    average_loss = epoch_loss / total_samples

    return (num_correct / num_samples * 100).item(), average_loss

def plot_layer_histograms_by_row(model, bins=50, count_nonzero_only=False):
    # 그룹 정의
    layer_groups = {
        "layer_1": [],
        "layer_2": [],
        "layer_3": [],
        "layer_4": [],
        "layer_5": [],
        "fc_layer": []
    }

    # 그룹화
    for name, param in model.named_parameters():
        if param.dim() > 1:  # Weight 파라미터만 고려
            if name.startswith("layer_1."):
                layer_groups["layer_1"].append((name, param))
            elif name.startswith("layer_2."):
                layer_groups["layer_2"].append((name, param))
            elif name.startswith("layer_3."):
                layer_groups["layer_3"].append((name, param))
            elif name.startswith("layer_4."):
                layer_groups["layer_4"].append((name, param))
            elif name.startswith("layer_5."):
                layer_groups["layer_5"].append((name, param))
            elif name.startswith("fc_layer."):
                layer_groups["fc_layer"].append((name, param))

    # Plot 설정
    num_rows = len(layer_groups)  # 각 layer 그룹별로 하나의 row
    max_cols = max(len(params) for params in layer_groups.values())  # 최대 컬럼 수
    fig, axes = plt.subplots(num_rows, max_cols, figsize=(4 * max_cols, 4 * num_rows), squeeze=False)

    # 각 그룹에 대해 히스토그램 생성
    for row_idx, (group_name, params) in enumerate(layer_groups.items()):
        for col_idx in range(max_cols):
            ax = axes[row_idx][col_idx]
            if col_idx < len(params):  # 현재 그룹의 파라미터가 있다면
                name, param = params[col_idx]
                param_cpu = param.detach().view(-1).cpu()
                if count_nonzero_only:
                    param_cpu = param_cpu[param_cpu != 0]
                ax.hist(param_cpu, bins=bins, density=True, alpha=0.5, color='blue')
                ax.set_title(name, fontsize=10)
                ax.set_xlabel("Weight Value")
                ax.set_ylabel("Density")
            else:  # 빈 subplot 처리
                ax.axis("off")
        axes[row_idx][0].set_ylabel(group_name, fontsize=12, labelpad=20)

    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    fig.suptitle("Weight Histograms by Layer", fontsize=16)
    plt.show()

def fine_grained_prune(tensor: torch.Tensor, sparsity : float) -> torch.Tensor:
    """
    magnitude-based pruning for single tensor
    :param tensor: torch.(cuda.)Tensor, weight of conv/fc layer
    :param sparsity: float, pruning sparsity
        sparsity = #zeros / #elements = 1 - #nonzeros / #elements
    :return:
        torch.(cuda.)Tensor, mask for zeros
    """
    sparsity = min(max(0.0, sparsity), 1.0)
    if sparsity == 1.0:
        tensor.zero_()
        return torch.zeros_like(tensor)
    elif sparsity == 0.0:
        return torch.ones_like(tensor)

    num_elements = tensor.numel()

    num_zeros = round(num_elements * sparsity)
    importance =  tensor.abs()
    threshold = importance.view(-1).kthvalue(num_zeros).values.item()
    mask =  importance > threshold

    tensor.mul_(mask)

    return mask

class FineGrainedPruner:
    def __init__(self, model, sparsity_dict):
        self.masks = FineGrainedPruner.prune(model, sparsity_dict)

    @torch.no_grad()
    def apply(self, model):
        for name, param in model.named_parameters():
            if name in self.masks:
                param *= self.masks[name]

    @staticmethod
    @torch.no_grad()
    def prune(model, sparsity_dict):
        masks = dict()
        for name, param in model.named_parameters():
            if param.dim() > 1: # we only prune conv and fc weights
                masks[name] = fine_grained_prune(param, sparsity_dict[name])
        return masks

def train(model: nn.Module, dataloader: DataLoader, criterion: nn.Module, optimizer: Optimizer, scheduler: LambdaLR, lambda_l1,lambda_l2,callbacks=None) -> None:
    model.train()
    epoch_loss = 0.0
    total_samples = 0
    for inputs, targets in tqdm(dataloader, desc='train', leave=False):
        # Move the data from CPU to GPU
        inputs = inputs.cuda()
        targets = targets.cuda()

        # Reset the gradients (from the last iteration)
        optimizer.zero_grad()

        # Forward inference
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        l1_loss = sum(torch.sum(torch.abs(param)) for param in model.parameters())
        l2_loss = sum(torch.sum(param ** 2) for param in model.parameters())

        loss = loss + lambda_l1 * l1_loss + lambda_l2 * l2_loss

        # Backward propagation
        loss.backward()

        # Update optimizer and LR scheduler
        optimizer.step()
        scheduler.step()

        batch_size = inputs.size(0)
        epoch_loss += loss.item() * batch_size
        total_samples += batch_size

        if callbacks is not None:
            for callback in callbacks:
                callback()
    average_loss = epoch_loss / total_samples

    return average_loss

pt_files = [file for file in os.listdir(".") if file.endswith(".pt")]
for file_name in pt_files:
    base_name = file_name[:-3]
    parts = base_name.split("_")
    lambda_l1 = float(parts[3])  # L1 뒤의 값
    lambda_l2 = float(parts[5])  # L2 뒤의 값

    wandb.init(
        project='Distributed Deep Learning_Pruning&Regularization',

        config={
            "learning_rate": 0.001,
            "architecture": "ResNet50",
            "dataset": "CIFAR10",
            "epochs":100,
            "lambda_l1":lambda_l1,
            "lambda_l2":lambda_l2
        }
    )
    wandb.run.name = f'L1_{lambda_l1} & L2_{lambda_l2} & weight_pruning'

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    image_size = 32
    transforms = {
        "train": Compose([
            RandomCrop(image_size, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
        ]),
        "test": ToTensor(),
    }
    dataset = {}
    for split in ["train", "test"]:
        dataset[split] = CIFAR10(
            root="data/cifar10",
            train=(split == "train"),
            download=True,
            transform=transforms[split],
        )

    dataloader = {}
    for split in ['train', 'test']:
        dataloader[split] = DataLoader(
            dataset[split],
            batch_size=512,
            shuffle=(split == 'train'),
            num_workers=0,
            pin_memory=True,
        )

    dataflow = {}
    for split in ['train', 'test']:
        dataflow[split] = DataLoader(
            dataset[split],
            batch_size=512,
            shuffle=(split == 'train'),
            num_workers=0,
            pin_memory=True,
        )

    model.load_state_dict(torch.load(file_name))
    num_finetune_epochs = 5
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_finetune_epochs)
    criterion = nn.CrossEntropyLoss()

    dense_model_accuracy, _ = evaluate(model, dataloader['test'], criterion)
    dense_model_size = get_model_size(model)
    print(f"dense model hias accuracy={dense_model_accuracy:.2f}%")
    print(f"dense model has size={dense_model_size / MiB:.2f} MiB")

    sparsity_dict = {}

    for name, param in model.named_parameters():
        sparsity_dict[name] = 0.5

    pruner = FineGrainedPruner(model, sparsity_dict)

    before_sparse_model_size = get_model_size(model, count_nonzero_only=True)
    print(f"Sparse model has size={before_sparse_model_size / MiB:.2f} MiB = {before_sparse_model_size / dense_model_size * 100:.2f}% of dense model size")
    before_sparse_model_accuracy, _ = evaluate(model, dataloader['test'], criterion)
    print(f"Sparse model has accuracy={before_sparse_model_accuracy:.2f}% before fintuning")

    best_sparse_model_checkpoint = dict()
    best_accuracy = 0
    print(f'Finetuning Fine-grained Pruned Sparse Model')
    for epoch in range(num_finetune_epochs):
        # At the end of each train iteration, we have to apply the pruning mask
        #    to keep the model sparse during the training
        train_loss = train(model, dataloader['train'], criterion, optimizer, scheduler,lambda_l1,lambda_l2,
              callbacks=[lambda: pruner.apply(model)])
        accuracy, test_loss = evaluate(model, dataloader['test'], criterion)
        wandb.log({"acc": accuracy, "train_loss": train_loss, "test_loss": test_loss})
        is_best = accuracy > best_accuracy
        if is_best:
            best_sparse_model_checkpoint['state_dict'] = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), f'best_acc_L1_{lambda_l1}_L2_{lambda_l2}_pruning.pt')
            best_accuracy = accuracy
        print(f'    Epoch {epoch+1} Accuracy {accuracy:.2f}% / Best Accuracy: {best_accuracy:.2f}%')

    # load the best sparse model checkpoint to evaluate the final performance
    model.load_state_dict(best_sparse_model_checkpoint['state_dict'])
    after_sparse_model_size = get_model_size(model, count_nonzero_only=True)
    print(f"Sparse model has size={after_sparse_model_size / MiB:.2f} MiB = {after_sparse_model_size / dense_model_size * 100:.2f}% of dense model size")
    after_sparse_model_accuracy, _ = evaluate(model, dataloader['test'], criterion)
    print(f"Sparse model has accuracy={after_sparse_model_accuracy:.2f}% after fintuning")
    wandb.log({"before_size": before_sparse_model_size / MiB,
               "before_size_percent": before_sparse_model_size / dense_model_size * 100,
               "before_acc": before_sparse_model_accuracy,
               "after_size": after_sparse_model_size / MiB,
               "after_size_percent": after_sparse_model_size / dense_model_size * 100,
               "after_acc": after_sparse_model_accuracy
               })
    wandb.finish()
