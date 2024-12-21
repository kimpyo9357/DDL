import os
import time
import torch
from torch import nn
from torchprofile import profile_macs
from matplotlib import pyplot as plt
# helper functions to measure latency of a regular PyTorch models.
#   Unlike fine-grained pruning, channel pruning
#   can directly leads to model size reduction and speed up.

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

def get_model_macs(model, inputs) -> int:
    return profile_macs(model, inputs)

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

model = ResNet().cuda()
pruned_model = ResNet().cuda()
@torch.no_grad()
def measure_latency(model, dummy_input, n_warmup=20, n_test=100):
    model.eval()
    # warmup
    for _ in range(n_warmup):
        _ = model(dummy_input)
    # real test
    t1 = time.time()
    for _ in range(n_test):
        _ = model(dummy_input)
    t2 = time.time()
    return (t2 - t1) / n_test  # average latency

def get_num_parameters(model: nn.Module, count_nonzero_only=True) -> int:
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

ori_path = "./saved/ori"
prun_path = "./saved/prun"
pt_files = [file for file in os.listdir(ori_path) if file.endswith(".pt")]
# for file_name in pt_files:
ori_file_name = pt_files[7]
prun_file_name = ori_file_name[:-3]+"_pruning"+ori_file_name[-3:]
print(ori_file_name)
print(prun_file_name)

model.load_state_dict(torch.load(os.path.join(ori_path,ori_file_name)))
pruned_model.load_state_dict(torch.load(os.path.join(prun_path,prun_file_name)))

# def plot_layer_histograms_by_row(title,model, bins=50, count_nonzero_only=False):
#     # 그룹 정의
#     layer_groups = {
#         "layer_1": [],
#         "layer_2": [],
#         "layer_3": [],
#         "layer_4": [],
#         "layer_5": [],
#         "fc_layer": []
#     }
#
#     # 그룹화
#     for name, param in model.named_parameters():
#         if param.dim() > 1:  # Weight 파라미터만 고려
#             if name.startswith("layer_1."):
#                 layer_groups["layer_1"].append((name, param))
#             elif name.startswith("layer_2."):
#                 layer_groups["layer_2"].append((name, param))
#             elif name.startswith("layer_3."):
#                 layer_groups["layer_3"].append((name, param))
#             elif name.startswith("layer_4."):
#                 layer_groups["layer_4"].append((name, param))
#             elif name.startswith("layer_5."):
#                 layer_groups["layer_5"].append((name, param))
#             elif name.startswith("fc_layer."):
#                 layer_groups["fc_layer"].append((name, param))
#
#     # Plot 설정
#     num_rows = len(layer_groups)  # 각 layer 그룹별로 하나의 row
#     max_cols = max(len(params) for params in layer_groups.values())  # 최대 컬럼 수
#     fig, axes = plt.subplots(num_rows, max_cols, figsize=(4 * max_cols, 4 * num_rows), squeeze=False)
#
#     # 각 그룹에 대해 히스토그램 생성
#     for row_idx, (group_name, params) in enumerate(layer_groups.items()):
#         for col_idx in range(max_cols):
#             ax = axes[row_idx][col_idx]
#             if col_idx < len(params):  # 현재 그룹의 파라미터가 있다면
#                 name, param = params[col_idx]
#                 param_cpu = param.detach().view(-1).cpu()
#                 if count_nonzero_only:
#                     param_cpu = param_cpu[param_cpu != 0]
#                 ax.hist(param_cpu, bins=bins, density=True, alpha=0.5, color='blue')
#                 ax.set_title(name, fontsize=10)
#                 ax.set_xlabel("Weight Value")
#                 ax.set_ylabel("Density")
#             else:  # 빈 subplot 처리
#                 ax.axis("off")
#         axes[row_idx][0].set_ylabel(group_name, fontsize=12, labelpad=20)
#
#     fig.tight_layout()
#     fig.subplots_adjust(top=0.95)
#     # fig.suptitle("Weight Histograms by Layer", fontsize=16)
#     fig.suptitle(title, fontsize=16)
#     plt.show()
# plot_layer_histograms_by_row("original",model, count_nonzero_only=True)
# plot_layer_histograms_by_row("pruned", pruned_model, count_nonzero_only=True)


table_template = "{:<15} {:<15} {:<15} {:<15}"
print (table_template.format('', 'Original','Pruned','Reduction Ratio'))

# 1. measure the latency of the original model and the pruned model on CPU
#   which simulates inference on an edge device
dummy_input = torch.randn(1, 3, 32, 32).to('cpu')
pruned_model = pruned_model.to('cpu')
model = model.to('cpu')

pruned_latency = measure_latency(pruned_model, dummy_input)
original_latency = measure_latency(model, dummy_input)
print(table_template.format('Latency (ms)',
                            round(original_latency * 1000, 1),
                            round(pruned_latency * 1000, 1),
                            round(original_latency / pruned_latency, 1)))

# 2. measure the computation (MACs)
original_macs = get_model_macs(model, dummy_input)
pruned_macs = get_model_macs(pruned_model, dummy_input)
print(table_template.format('MACs (M)',
                            round(original_macs / 1e6),
                            round(pruned_macs / 1e6),
                            round(original_macs / pruned_macs, 1)))

# 3. measure the model size (params)
original_param = int(get_num_parameters(model))
pruned_param = int(get_num_parameters(pruned_model))
print(table_template.format('Param (M)',
                            round(original_param / 1e6, 2),
                            round(pruned_param/ 1e6, 2),
                            round(original_param / pruned_param, 1)))

# # put model back to cuda
# pruned_model = pruned_model.to('cuda')
# model = model.to('cuda')