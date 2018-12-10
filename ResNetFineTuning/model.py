import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class PretrainedInception(nn.Module):
    def __init__(self):
        super(PretrainedInception, self).__init__()
        self.four_to_three_dim = nn.Conv2d(4, 3, 1)
        self.pretrained = torchvision.models.inception_v3(pretrained='imagenet')
        self.out_size = self.pretrained.fc.in_features
        self.pretrained.fc = Identity()

        self.final_layer = nn.Linear(self.out_size, 28)
        self.final_activ = nn.Sigmoid()

    def forward(self, inputs):
        x = self.four_to_three_dim(inputs)
        x = self.pretrained(x)
        x = self.final_layer(x)
        x = self.final_activ(x)
        return x


class PretrainedResnet152(nn.Module):
    def __init__(self):
        super(PretrainedResnet152, self).__init__()
        self.pretrained = torchvision.models.resnet152(pretrained='imagenet')
        self.out_size = self.pretrained.fc.in_features
        self.pretrained.fc = Identity()
        self.pretrained.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.final_layer = nn.Linear(self.out_size, 2048)
        self.final_norm = nn.BatchNorm1d(2048)
        self.final_activ = nn.PReLU(2048)
        self.final_layer_2 = nn.Linear(2048, 28)
        self.final_activ_2 = nn.Sigmoid()

    def forward(self, inputs):
        x = self.pretrained(inputs)
        x = self.final_layer(x)
        x = self.final_norm(x)
        x = self.final_activ(x)
        x = self.final_layer_2(x)
        x = self.final_activ_2(x)
        return x


class PretrainedResnet50(nn.Module):
    def __init__(self):
        super(PretrainedResnet50, self).__init__()
        self.pretrained = torchvision.models.resnet50(pretrained='imagenet')
        self.out_size = self.pretrained.fc.in_features
        self.pretrained.fc = Identity()
        conv_1_weights = self.pretrained.conv1.weight.data
        self.pretrained.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.pretrained.conv1.weight.data = torch.zeros(size=self.pretrained.conv1.weight.data.shape)
        self.pretrained.conv1.weight.data[:, :3] = conv_1_weights

        self.pen_norm = nn.BatchNorm1d(self.out_size)
        self.pen_layer = nn.Linear(self.out_size, 2048)
        self.final_norm = nn.BatchNorm1d(2048)
        self.final_activ = nn.PReLU(2048)
        self.final_layer = nn.Linear(2048, 28)

    def forward(self, inputs):
        x = self.pretrained(inputs)
        x = self.pen_norm(x)
        x = F.dropout(x, 0.5)
        x = self.pen_layer(x)
        x = self.final_activ(x)
        x = self.final_norm(x)
        x = F.dropout(x, 0.5)
        x = self.final_layer(x)
        return x

class PretrainedZoo(nn.Module):
    def __init__(self):
        super(PretrainedZoo, self).__init__()
        self.pretrained_resnet152 = torchvision.models.resnet152(pretrained='imagenet')
        self.out_size_1 = self.pretrained_resnet152.fc.in_features
        self.pretrained_resnet152.fc = Identity()
        self.pretrained_resnet152.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.pretrained_resnet18 = torchvision.models.resnet18(pretrained='imagenet')
        self.out_size_2 = self.pretrained_resnet18.fc.in_features
        self.pretrained_resnet18.fc = Identity()
        self.pretrained_resnet18.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.pretrained_densenet201 = torchvision.models.densenet201(pretrained='imagenet')
        self.out_size_3 = self.pretrained_densenet201.classifier.in_features
        self.pretrained_densenet201.classifier = Identity()
        self.pretrained_densenet201.features[0] = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.pretrained_squeezenet1_1 = torchvision.models.squeezenet1_1(pretrained='imagenet')
        self.out_size_4 = 512
        self.pretrained_squeezenet1_1.classifier = Identity()
        self.pretrained_squeezenet1_1.features[0] = nn.Conv2d(4, 64, kernel_size=7, stride=2)

        self.final_layer = nn.Linear(self.out_size_1 + self.out_size_2 + self.out_size_3 + self.out_size_4, 2048)
        self.final_activ = nn.PReLU(2048)
        self.final_layer_2 = nn.Linear(2048, 28)
        self.final_sig = nn.Sigmoid()

    def forward(self, x):
        y = self.pretrained_resnet152(x)
        z = self.pretrained_densenet201(x)
        w = self.pretrained_resnet18(x)
        v = self.pretrained_squeezenet1_1.features(x)
        v = F.avg_pool2d(v, 13, stride=1)
        v = v.view(-1, 512)

        x = torch.cat([y, z, w, v], dim=1)

        x = F.dropout(x, 0.5)
        x = self.final_layer(x)
        x = self.final_activ(x)
        x = F.dropout(x, 0.5)
        x = self.final_layer_2(x)
        x = self.final_sig(x)
        return x


class ResNet18(nn.Module):
    def __init__(self, hidden_size=64, output_size=28, in_channels=4, blocks=4):
        super(ResNet18, self).__init__()
        self.hidden_size = hidden_size
        self.blocks = blocks

        self.initial_conv = nn.Conv2d(in_channels, hidden_size, 7, padding=3)
        self.initial_norm = nn.BatchNorm2d(hidden_size)
        self.initial_activ = nn.PReLU(hidden_size)

        self.convs1 = nn.ModuleList(
            [nn.Conv2d(hidden_size * 2 ** i, hidden_size * 2 ** i, 3, padding=1) for
             i in range(blocks)])
        self.norm1 = nn.ModuleList([nn.BatchNorm2d(hidden_size * (2 ** i)) for i in range(blocks)])
        self.activ1 = nn.ModuleList([nn.PReLU(hidden_size * (2 ** i)) for i in range(blocks)])

        self.convs2 = nn.ModuleList(
            [nn.Conv2d(hidden_size * 2 ** i, hidden_size * 2 ** i, 3, padding=1) for
             i in range(blocks)])
        self.norm2 = nn.ModuleList([nn.BatchNorm2d(hidden_size * (2 ** i)) for i in range(blocks)])
        self.activ2 = nn.ModuleList([nn.PReLU(hidden_size * (2 ** i)) for i in range(blocks)])


        self.convs3 = nn.ModuleList(
            [nn.Conv2d(hidden_size * 2 ** i, hidden_size * 2 ** i, 3, padding=1) for
             i in range(blocks)])
        self.norm3 = nn.ModuleList([nn.BatchNorm2d(hidden_size * (2 ** i)) for i in range(blocks)])
        self.activ3 = nn.ModuleList([nn.PReLU(hidden_size * (2 ** i)) for i in range(blocks)])

        self.convs4 = nn.ModuleList(
            [nn.Conv2d(hidden_size * 2 ** i, hidden_size * 2 ** i, 3, padding=1) for
             i in range(blocks)])
        self.norm4 = nn.ModuleList([nn.BatchNorm2d(hidden_size * (2 ** i)) for i in range(blocks)])
        self.activ4 = nn.ModuleList([nn.PReLU(hidden_size * (2 ** i)) for i in range(blocks)])


        self.transitions_conv = nn.ModuleList(
            [nn.Conv2d(hidden_size * 2 ** i, hidden_size * 2 ** (i+1), 3, padding=1) for
             i in range(blocks)])
        self.transitions_norm = nn.ModuleList([nn.BatchNorm2d(hidden_size * 2 ** (i + 1)) for i in range(blocks)])
        self.transitions_activ = nn.ModuleList([nn.PReLU(hidden_size * 2 ** (i + 1)) for i in range(blocks)])

        final_size = hidden_size * 2 ** blocks
        self.final_linear1 = nn.Linear(final_size, final_size)
        self.final_norm1 = nn.BatchNorm1d(final_size)
        self.final_activ1 = nn.PReLU(final_size)
        self.final_linear2 = nn.Linear(final_size, output_size)
        self.final_activ2 = nn.Sigmoid()

    def forward(self, inputs):
        x = self.initial_conv(inputs)
        x = self.initial_norm(x)
        x = self.initial_activ(x)

        for i in range(self.blocks):
            fx = self.convs1[i](x)
            fx = self.norm1[i](fx)
            fx = self.activ1[i](fx)
            fx = self.convs2[i](fx)
            fx = self.norm2[i](fx)
            fx = self.activ2[i](fx)

            x = x + fx

            fx = self.convs3[i](x)
            fx = self.norm3[i](fx)
            fx = self.activ3[i](fx)
            fx = self.convs4[i](fx)
            fx = self.norm4[i](fx)
            fx = self.activ4[i](fx)

            x = x + fx

            x = self.transitions_conv[i](x)
            x = self.transitions_norm[i](x)
            x = self.transitions_activ[i](x)

            x = F.max_pool2d(x, kernel_size=2)
        x = F.avg_pool2d(x, kernel_size=(x.shape[-2], x.shape[-1]))
        x = x.view(x.shape[0], -1)

        x = self.final_linear1(x)
        x = self.final_norm1(x)
        x = self.final_activ1(x)
        x = self.final_linear2(x)
        x = self.final_activ2(x)

        return x


if __name__== '__main__':
    m = PretrainedResnet50()
    for param in m.pretrained.conv1.parameters():
        param.requires_grad=False
    for param in m.pretrained.parameters():
        param.requires_grad = False
        print(param)