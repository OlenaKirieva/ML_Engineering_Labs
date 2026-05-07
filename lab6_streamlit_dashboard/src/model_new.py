import torch
import torch.nn as nn


class CIFAR10ProCNN(nn.Module):
    def __init__(self, n_classes: int = 10) -> None:
        super(CIFAR10ProCNN, self).__init__()

        def conv_block(in_f, out_f, dropout_rate):
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_f),
                nn.ReLU(),
                nn.Conv2d(out_f, out_f, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_f),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout(dropout_rate),
            )

        self.block1 = conv_block(3, 32, 0.2)  # 32x32 -> 16x16
        self.block2 = conv_block(32, 64, 0.3)  # 16x16 -> 8x8
        self.block3 = conv_block(64, 128, 0.4)  # 8x8 -> 4x4
        self.block4 = conv_block(128, 256, 0.5)  # 4x4 -> 2x2

        self.flat = nn.Flatten()
        self.fc = nn.Linear(256 * 2 * 2, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.flat(x)
        x = self.fc(x)
        return x
