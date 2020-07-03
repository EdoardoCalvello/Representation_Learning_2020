import torch.nn.functional as F
from math import sqrt
# from functions import onehot

import os
from os.path import join


class Discriminator(nn.Module):
    def __init__(self, feat_dim=128):
        super().__init__()

        # ConvBlock(in_channel, out_channel, kernel_size, padding, kernel_size2=None, padding2=None,pixel_norm=True)
        self.progression = nn.ModuleList([ConvBlock(feat_dim // 4, feat_dim // 4, 3, 1),
                                          ConvBlock(feat_dim // 4, feat_dim // 2, 3, 1),
                                          ConvBlock(feat_dim // 2, feat_dim, 3, 1),
                                          ConvBlock(feat_dim, feat_dim, 3, 1),
                                          ConvBlock(feat_dim, feat_dim, 3, 1),
                                          ConvBlock(feat_dim, feat_dim, 3, 1),
                                          ConvBlock(feat_dim + 1, feat_dim, 3, 1, 4, 0)])

        self.from_rgb = nn.ModuleList([EqualConv2d(3, feat_dim // 4, 1),
                                       EqualConv2d(3, feat_dim // 4, 1),
                                       EqualConv2d(3, feat_dim // 2, 1),
                                       EqualConv2d(3, feat_dim, 1),
                                       EqualConv2d(3, feat_dim, 1),
                                       EqualConv2d(3, feat_dim, 1),
                                       EqualConv2d(3, feat_dim, 1)])

        self.n_layer = len(self.progression)

        self.linear = EqualLinear(feat_dim, 1)

    def forward(self, input, step=0, alpha=-1):
        for i in range(step, -1, -1):
            index = self.n_layer - i - 1

            if i == step:
                out = self.from_rgb[index](input)

            if i == 0:
                out_std = torch.sqrt(out.var(0, unbiased=False) + 1e-8)
                mean_std = out_std.mean()
                mean_std = mean_std.expand(out.size(0), 1, 4, 4)
                out = torch.cat([out, mean_std], 1)

            out = self.progression[index](out)

            if i > 0:
                # out = F.avg_pool2d(out, 2)
                out = F.interpolate(out, scale_factor=0.5, mode='bilinear', align_corners=False)

                if i == step and 0 <= alpha < 1:
                    # skip_rgb = F.avg_pool2d(input, 2)
                    skip_rgb = F.interpolate(input, scale_factor=0.5, mode='bilinear', align_corners=False)
                    skip_rgb = self.from_rgb[index + 1](skip_rgb)
                    out = (1 - alpha) * skip_rgb + alpha * out

        out = out.squeeze(2).squeeze(2)
        # print(input.size(), out.size(), step)
        out = self.linear(out)

        return out

    def save_params(self, exDir):
        print('saving params...')
        torch.save(self.state_dict(), join(exDir, 'class_dis_params'))

    def load_params(self, exDir):
        print('loading params...')
        self.load_state_dict(torch.load(join(exDir, 'class_dis_params')))


class DISCRIMINATOR(nn.Module):

    def __init__(self, imSize, fSize=64, numLabels=1):
        super(DISCRIMINATOR, self).__init__()
        # define layers here

        self.fSize = fSize
        self.imSize = imSize

        inSize = imSize // (2 ** 4)
        self.numLabels = numLabels

        self.dis1 = nn.Conv2d(3, fSize, 5, stride=2, padding=2)
        self.dis2 = nn.Conv2d(fSize, fSize * 2, 5, stride=2, padding=2)
        self.dis3 = nn.Conv2d(fSize * 2, fSize * 4, 5, stride=2, padding=2)
        self.dis4 = nn.Conv2d(fSize * 4, fSize * 8, 5, stride=2, padding=2)
        self.dis5 = nn.Linear((fSize * 8) * inSize * inSize, numLabels)

        self.useCUDA = torch.cuda.is_available()

    def discriminate(self, x):
        x = F.relu(self.dis1(x))
        x = F.relu(self.dis2(x))
        x = F.relu(self.dis3(x))
        x = F.relu(self.dis4(x))
        x = x.view(x.size(0), -1)
        if self.numLabels == 1:
            x = torch.sigmoid(self.dis5(x))
        else:
            x = F.softmax(self.dis5(x))

        return x

    def forward(self, x):
        # the outputs needed for training
        return self.discriminate(x)


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True)
                                  + 1e-8)


class CVAE1(nn.Module):

    def __init__(self, nz, imSize, fSize=64, sig=1):
        super(CVAE1, self).__init__()
        # define layers here

        self.fSize = fSize
        self.nz = nz
        self.imSize = imSize
        self.sig = sig

        inSize = imSize // (2 ** 4)
        self.inSize = inSize

        self.enc1 = nn.Conv2d(3, fSize, 5, stride=2, padding=2)
        self.enc2 = nn.Conv2d(fSize, fSize * 2, 5, stride=2, padding=2)
        self.enc3 = nn.Conv2d(fSize * 2, fSize * 4, 5, stride=2, padding=2)
        self.enc4 = nn.Conv2d(fSize * 4, fSize * 8, 5, stride=2, padding=2)

        self.encLogVar = nn.Linear((fSize * 8) * inSize * inSize, nz)
        self.encMu = nn.Linear((fSize * 8) * inSize * inSize, nz)
        self.encY = nn.Linear((fSize * 8) * inSize * inSize, 1)

        self.dec1 = nn.Linear(nz + 1, (fSize * 8) * inSize * inSize)
        self.dec2 = nn.ConvTranspose2d(fSize * 8, fSize * 4, 3, stride=2, padding=1, output_padding=1)
        self.dec2b = nn.BatchNorm2d(fSize * 4)
        self.dec3 = nn.ConvTranspose2d(fSize * 4, fSize * 2, 3, stride=2, padding=1, output_padding=1)
        self.dec3b = nn.BatchNorm2d(fSize * 2)
        self.dec4 = nn.ConvTranspose2d(fSize * 2, fSize, 3, stride=2, padding=1, output_padding=1)
        self.dec4b = nn.BatchNorm2d(fSize)
        self.dec5 = nn.ConvTranspose2d(fSize, 3, 3, stride=2, padding=1, output_padding=1)

        self.useCUDA = torch.cuda.is_available()

    def encode(self, x):
        # define the encoder here return mu(x) and sigma(x)
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        x = F.relu(self.enc4(x))
        x = x.view(x.size(0), -1)
        mu = self.encMu(x)  # no relu - mean may be negative
        log_var = self.encLogVar(x)  # no relu - log_var may be negative
        y = torch.sigmoid(self.encY(x.detach()))

        return mu, log_var, y

    def re_param(self, mu, log_var):
        # do the re-parameterising here
        sigma = torch.exp(log_var / 2)  # sigma = exp(log_var/2) #torch.exp(log_var/2)
        if self.useCUDA:
            eps = torch.randn(sigma.size(0), self.nz).cuda()
        else:
            eps = torch.randn(sigma.size(0), self.nz)

        return mu + sigma * eps  # eps.mul(simga)._add(mu)

    def sample_z(self, noSamples, sig=1):
        z = sig * torch.randn(noSamples, self.nz)
        if self.useCUDA:
            return z.cuda()
        else:
            return z

    def decode(self, y, z):
        # define the decoder here
        z = torch.cat([y, z], dim=1)
        z = F.relu(self.dec1(z))
        z = z.view(z.size(0), -1, self.inSize, self.inSize)
        z = F.relu(self.dec2b(self.dec2(z)))
        z = F.relu(self.dec3b(self.dec3(z)))
        z = F.relu(self.dec4b(self.dec4(z)))
        z = torch.sigmoid(self.dec5(z))

        return z

    def forward(self, x):
        # the outputs needed for training
        mu, log_var, y = self.encode(x)
        z = self.re_param(mu, log_var)
        reconstruction = self.decode(y, z)

        return reconstruction, mu, log_var, y

    def save_params(self, exDir):
        print('saving params...')
        torch.save(self.state_dict(), join(exDir, 'cvae1_params'))

    def load_params(self, exDir):
        print('loading params...')
        self.load_state_dict(torch.load(join(exDir, 'cvae1_params')))

    def loss(self, rec_x, x, mu, logVar):
        sigma2 = torch.Tensor([self.sig])
        if self.useCUDA:
            sigma2 = sigma2.cuda()
        logVar2 = torch.log(sigma2)
        # Total loss is BCE(x, rec_x) + KL
        BCE = F.binary_cross_entropy(rec_x, x,
                                     size_average=False)  # not averaged over mini-batch if size_average=FALSE and is averaged if =True
        # (might be able to use nn.NLLLoss2d())
        if self.sig == 1:
            KL = 0.5 * torch.sum(mu ** 2 + torch.exp(logVar) - 1. - logVar)  # 0.5 * sum(1 + log(var) - mu^2 - var)
        else:
            KL = 0.5 * torch.sum(logVar2 - logVar + torch.exp(logVar) + (mu ** 2 / 2 * sigma2 ** 2) - 0.5)
        return BCE / (x.size(2) ** 2), KL / mu.size(1)


class AUX(nn.Module):
    # map z to a label, y
    def __init__(self, nz, numLabels=1):
        super(AUX, self).__init__()

        self.nz = nz
        self.numLabels = numLabels

        self.aux1 = nn.Linear(nz, 1000)
        self.aux2 = nn.Linear(1000, 1000)
        self.aux3 = nn.Linear(1000, numLabels)

    def infer_y_from_z(self, z):
        z = F.relu(self.aux1(z))
        z = F.relu(self.aux2(z))
        if self.numLabels == 1:
            z = torch.sigmoid(self.aux3(z))
        else:
            z = F.softmax(self.aux3(z))

        return z

    def forward(self, z):
        return self.infer_y_from_z(z)

    def loss(self, pred, target):
        return F.nll_loss(pred, target)

    def save_params(self, exDir):
        print('saving params...')
        torch.save(self.state_dict(), join(exDir, 'aux_params'))

    def load_params(self, exDir):
        print('loading params...')
        self.load_state_dict(torch.load(join(exDir, 'aux_params')))


######################################################################################################################################
######################################################################################################################################
######################################################################################################################################
######################################################################################################################################
######################################################################################################################################
######################################################################################################################################
######################################################################################################################################
######################################################################################################################################
######################################################################################################################################
######################################################################################################################################

class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module


class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)


class EqualConvTranspose2d(nn.Module):
    ### additional module for OOGAN usage
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.ConvTranspose2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding, kernel_size2=None, padding2=None,
                 pixel_norm=True):
        super().__init__()

        pad1 = padding
        pad2 = padding
        if padding2 is not None:
            pad2 = padding2

        kernel1 = kernel_size
        kernel2 = kernel_size
        if kernel_size2 is not None:
            kernel2 = kernel_size2

        convs = [EqualConv2d(in_channel, out_channel, kernel1, padding=pad1)]
        if pixel_norm:
            convs.append(PixelNorm())
        convs.append(nn.LeakyReLU(0.1))
        convs.append(EqualConv2d(out_channel, out_channel, kernel2, padding=pad2))
        if pixel_norm:
            convs.append(PixelNorm())
        convs.append(nn.LeakyReLU(0.1))

        self.conv = nn.Sequential(*convs)

    def forward(self, input):
        out = self.conv(input)
        return out


def upscale(feat):
    return F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=False)


class Generator(nn.Module):
    def __init__(self, input_code_dim=128, in_channel=128, pixel_norm=True, tanh=True):
        super().__init__()
        self.input_dim = input_code_dim
        self.tanh = tanh
        self.input_layer = nn.Sequential(
            EqualConvTranspose2d(input_code_dim, in_channel, 4, 1, 0),
            PixelNorm(),
            nn.LeakyReLU(0.1))

        self.progression_4 = ConvBlock(in_channel, in_channel, 3, 1, pixel_norm=pixel_norm)
        self.progression_8 = ConvBlock(in_channel, in_channel, 3, 1, pixel_norm=pixel_norm)
        self.progression_16 = ConvBlock(in_channel, in_channel, 3, 1, pixel_norm=pixel_norm)
        self.progression_32 = ConvBlock(in_channel, in_channel, 3, 1, pixel_norm=pixel_norm)
        self.progression_64 = ConvBlock(in_channel, in_channel // 2, 3, 1, pixel_norm=pixel_norm)
        self.progression_128 = ConvBlock(in_channel // 2, in_channel // 4, 3, 1, pixel_norm=pixel_norm)
        self.progression_256 = ConvBlock(in_channel // 4, in_channel // 4, 3, 1, pixel_norm=pixel_norm)

        self.to_rgb_8 = EqualConv2d(in_channel, 3, 1)
        self.to_rgb_16 = EqualConv2d(in_channel, 3, 1)
        self.to_rgb_32 = EqualConv2d(in_channel, 3, 1)
        self.to_rgb_64 = EqualConv2d(in_channel // 2, 3, 1)
        self.to_rgb_128 = EqualConv2d(in_channel // 4, 3, 1)
        self.to_rgb_256 = EqualConv2d(in_channel // 4, 3, 1)

        self.max_step = 6

    def progress(self, feat, module):
        out = F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=False)
        out = module(out)
        return out

    def output(self, feat1, feat2, module1, module2, alpha):
        if 0 <= alpha < 1:
            skip_rgb = upscale(module1(feat1))
            out = (1 - alpha) * skip_rgb + alpha * module2(feat2)
        else:
            out = module2(feat2)
        if self.tanh:
            return torch.tanh(out)
        return out

    def forward(self, input, step=0, alpha=-1):
        if step > self.max_step:
            step = self.max_step

        out_4 = self.input_layer(input.view(-1, self.input_dim, 1, 1))
        out_4 = self.progression_4(out_4)
        out_8 = self.progress(out_4, self.progression_8)
        if step == 1:
            if self.tanh:
                return torch.tanh(self.to_rgb_8(out_8))
            return self.to_rgb_8(out_8)

        out_16 = self.progress(out_8, self.progression_16)
        if step == 2:
            return self.output(out_8, out_16, self.to_rgb_8, self.to_rgb_16, alpha)

        out_32 = self.progress(out_16, self.progression_32)
        if step == 3:
            return self.output(out_16, out_32, self.to_rgb_16, self.to_rgb_32, alpha)

        out_64 = self.progress(out_32, self.progression_64)
        if step == 4:
            return self.output(out_32, out_64, self.to_rgb_32, self.to_rgb_64, alpha)

        out_128 = self.progress(out_64, self.progression_128)
        if step == 5:
            return self.output(out_64, out_128, self.to_rgb_64, self.to_rgb_128, alpha)

        out_256 = self.progress(out_128, self.progression_256)
        if step == 6:
            return self.output(out_128, out_256, self.to_rgb_128, self.to_rgb_256, alpha)


