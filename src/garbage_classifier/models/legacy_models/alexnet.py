"""AlexNet (2012) - 第一个现代 CNN，ImageNet 冠军。

学习要点：
  - 5 个卷积层 + 3 个全连接层，是"卷积提特征 + 全连接分类"的经典范式。
  - 用 ReLU（而非 tanh）缓解梯度消失；用 Dropout 防过拟合。
  - 历史意义大于实用价值：结构已被 ResNet 等超越，但思想仍是所有 CNN 的起点。
"""

# @Author  : James
# @File    : alexnet.py
# @Description :
import torch
import torch.nn as nn
try:
    from ptflops import get_model_complexity_info
except ImportError:
    get_model_complexity_info = None
try:
    from torchinfo import summary
except ImportError:
    summary = None

__all__ = [
    'AlexNet',
]

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),          # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),   # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),          # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),  # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),  # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),          # output[128, 6, 6]
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(128 * 6 * 6, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # 一般默认使用kaiming初始化方法
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)  # 正态分布初始化
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    net = AlexNet(num_classes=6)
    # print(net)
    summary(net, input_size=(1, 3, 224, 224))
    # alexnet = models.AlexNet()
    # print(alexnet)
    macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
