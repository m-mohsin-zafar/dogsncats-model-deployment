from torch.nn import AvgPool2d, Conv2d, Linear, Sequential, MaxPool2d, \
    ReLU, Module, BatchNorm2d, Dropout


class TenLayerConvNet(Module):
    def __init__(self, num_classes=2):
        super(TenLayerConvNet, self).__init__()
        self.conv_1 = Sequential(
            Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=2),
            BatchNorm2d(8),
            ReLU()
        )
        self.conv_2 = Sequential(
            Conv2d(8, 16, kernel_size=3, stride=1, padding=2),
            BatchNorm2d(16),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2)
        )
        self.dropout_1 = Sequential(
            Dropout(0.3)
        )
        self.conv_3 = Sequential(
            Conv2d(16, 32, kernel_size=3, stride=1),
            BatchNorm2d(32),
            ReLU()
        )
        self.dropout_2 = Sequential(
            Dropout(0.1)
        )
        self.conv_4 = Sequential(
            Conv2d(32, 48, kernel_size=1, stride=1),
            BatchNorm2d(48),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2)
        )
        self.dropout_3 = Sequential(
            Dropout(0.1)
        )
        self.conv_5 = Sequential(
            Conv2d(48, 48, kernel_size=1, stride=1),
            BatchNorm2d(48),
            ReLU(),
        )
        self.conv_6 = Sequential(
            Conv2d(48, 64, kernel_size=3, stride=2),
            BatchNorm2d(64),
            ReLU(),
            MaxPool2d(kernel_size=2, stride=2)
        )
        self.dropout_4 = Sequential(
            Dropout(0.3)
        )
        self.conv_7 = Sequential(
            Conv2d(64, 96, kernel_size=3, stride=1),
            BatchNorm2d(96),
            ReLU()
        )
        self.conv_8 = Sequential(
            Conv2d(96, 112, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(112),
            ReLU()
        )
        self.conv_9 = Sequential(
            Conv2d(112, 128, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(128),
            ReLU()
        )
        self.dropout_5 = Sequential(
            Dropout(0.4)
        )
        self.conv_10 = Sequential(
            Conv2d(128, 160, kernel_size=3, stride=1),
            BatchNorm2d(160),
            ReLU(),
            AvgPool2d(kernel_size=2, stride=2)
        )
        self.fc = Linear(3 * 3 * 160, num_classes)

    def forward(self, x):
        # Shape = (Batch_size, 3, 192, 192)
        out = self.conv_1(x)
        # Shape = (Batch_size, 8, 194, 194)
        out = self.conv_2(out)
        # Shape = (Batch_size, 16, 98, 98)
        out = self.dropout_1(out)
        out = self.conv_3(out)
        # Shape = (Batch_size, 32, 96, 96)
        out = self.dropout_2(out)
        out = self.conv_4(out)
        # Shape = (Batch_size, 48, 48, 48)
        out = self.dropout_3(out)
        out = self.conv_5(out)
        # Shape = (Batch_size, 48, 48, 48)
        out = self.conv_6(out)
        out = self.dropout_4(out)
        # Shape = (Batch_size, 64, 11, 11)
        out = self.conv_7(out)
        # Shape = (Batch_size, 96, 9, 9)
        out = self.conv_8(out)
        # Shape = (Batch_size, 112, 9, 9)
        out = self.conv_9(out)
        # Shape = (Batch_size, 128, 9, 9)
        out = self.dropout_5(out)
        out = self.conv_10(out)
        # Shape = (Batch_size, 160, 3, 3)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
