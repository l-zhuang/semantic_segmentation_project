import torch
import torch.nn as nn


class InitialBlock(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 bias=False,
                 relu=True):
        super().__init__()

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        self.main_branch = nn.Conv2d(
            in_channels,
            out_channels - 3,
            kernel_size = 3,
            stride = 2,
            padding = 1,
            bias=bias)

        self.ext_branch = nn.MaxPool2d(3, stride = 2, padding = 1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.out_activation = activation()

    def forward(self, x):
        main = self.main_branch(x)
        ext = self.ext_branch(x)
        out = torch.cat((main, ext), 1)
        out = self.batch_norm(out)
        output = self.out_activation(out)
        return output


class RegularBottleneck(nn.Module):

    def __init__(self,
                 channels,
                 internal_ratio=4,
                 kernel_size=3,
                 padding=0,
                 dilation=1,
                 asymmetric=False,
                 dropout_prob=0,
                 bias=False,
                 relu=True):
        super().__init__()

        if internal_ratio <= 1 or internal_ratio > channels:
            raise RuntimeError("Value out of range. Expected value in the "
                               "interval [1, {0}], got internal_scale={1}."
                               .format(channels, internal_ratio))
        internal_channels = channels // internal_ratio

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU
        
        # Main branch #

        # 1x1 projection convolution
        self.main_conv_block_1 = nn.Sequential(
            nn.Conv2d(
                channels,
                internal_channels,
                kernel_size=1,
                stride=1,
                bias=bias), 
                nn.BatchNorm2d(internal_channels), 
                activation())

        # If the convolution is asymmetric we split the main convolution
        # the first is 5x1 and the second is 1x5.
        if asymmetric:
            self.main_conv_block_2  = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(kernel_size, 1),
                    stride=1,
                    padding=(padding, 0),
                    dilation=dilation,
                    bias=bias), 
                    nn.BatchNorm2d(internal_channels), 
                    activation(),
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=(1, kernel_size),
                    stride=1,
                    padding=(0, padding),
                    dilation=dilation,
                    bias=bias), 
                    nn.BatchNorm2d(internal_channels), 
                    activation())
        else:
            self.main_conv_block_2 = nn.Sequential(
                nn.Conv2d(
                    internal_channels,
                    internal_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=bias), 
                    nn.BatchNorm2d(internal_channels), 
                    activation())

        # 1x1 expansion convolution
        self.main_conv_block_3 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                channels,
                kernel_size=1,
                stride=1,
                bias=bias), 
                nn.BatchNorm2d(channels), 
                activation())
        
        # Dropout Regularization
        self.dropout = nn.Dropout2d(p=dropout_prob)

        # PReLU layer to apply after adding the branches
        self.out_activation = activation()

    def forward(self, x):
        # Main branch shortcut
        main = x

        # Extension branch
        branch = self.main_conv_block_1(x)
        branch = self.main_conv_block_2(branch)
        branch = self.main_conv_block_3(branch)
        branch = self.dropout(branch)

        # Add main and extension branches
        out = main + branch
        # activation layer
        output = self.out_activation(out)

        return output


class DownsamplingBottleneck(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 internal_ratio=4,
                 return_indices=False,
                 dropout_prob=0,
                 bias=False,
                 relu=True):
        super().__init__()
        internal_channels = in_channels // internal_ratio
        self.return_indices = return_indices

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        # MAIN
        # Conv 2X2
        self.main_conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                internal_channels,
                kernel_size=2,
                stride=2,
                bias=bias), 
                nn.BatchNorm2d(internal_channels), 
                activation())

        # Conv 3x3
        self.main_conv_block_2 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                internal_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias), 
                nn.BatchNorm2d(internal_channels), 
                activation())

        # Conv 1x1
        self.main_conv_block_3 = nn.Sequential(
            nn.Conv2d(
                internal_channels,
                out_channels,
                kernel_size=1,
                stride=1,
                bias=bias), 
                nn.BatchNorm2d(out_channels), 
                activation())

        # branch #
        # maxpooling
        self.branch = nn.MaxPool2d(
            2,
            stride=2,
            return_indices=return_indices)
       
        # dropout layer
        self.dropout = nn.Dropout2d(p=dropout_prob)

        # PReLU layer to apply after concatenating the branches
        self.out_activation = activation()

    def forward(self, x):

        # Main
        main = self.main_conv_block_1(x)
        main = self.main_conv_block_2(main)
        main = self.main_conv_block_3(main)
        main = self.dropout(main)
        # Branch
        if self.return_indices:
            shortcut, max_indices = self.branch(x)
        else:
            shortcut = self.branch(x)

        # padding
        n, ch_main, h, w = main.size()
        ch_branch = shortcut.size()[1]
        padding = torch.zeros(n, ch_main-ch_branch, h, w)

        # Before concatenating, check if main is on the CPU or GPU and
        if shortcut.is_cuda:
            padding = padding.cuda()
  
        # Concatenate
        shortcut = torch.cat((shortcut, padding), 1)

        # Add main and extension branches
        out = main + shortcut
        output = self.out_activation(out), max_indices
        
        return output, max_indices


class UpsamplingBottleneck(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 internal_ratio=4,
                 dropout_prob=0,
                 bias=False,
                 relu=True):
        super().__init__()

        internal_channels = in_channels // internal_ratio

        if relu:
            activation = nn.ReLU
        else:
            activation = nn.PReLU

        # branch
        self.branch = nn.Sequential(
            nn.Conv2d(in_channels, 
                      out_channels, 
                      kernel_size=1, 
                      bias=bias),
            nn.BatchNorm2d(out_channels))

        # the max pooling layers
        self.branch_unpool1 = nn.MaxUnpool2d(kernel_size=2)

        # Main #
        # Conv 1x1
        self.main_conv_1 = nn.Sequential(
            nn.Conv2d(
                in_channels, 
                internal_channels, 
                kernel_size=1, 
                bias=bias),
                nn.BatchNorm2d(internal_channels), 
                activation())

        # Transposed convolution
        self.main_transpose_2 = nn.ConvTranspose2d(
            internal_channels,
            internal_channels,
            kernel_size=2,
            stride=2,
            bias=bias)
        self.main_bn_2 = nn.BatchNorm2d(internal_channels)
        self.main_act_2 = activation()

        # Conv 1x1
        self.main_conv_3 = nn.Sequential(
            nn.Conv2d(
                internal_channels, 
                out_channels, 
                kernel_size=1, 
                bias=bias),
            nn.BatchNorm2d(out_channels))

        self.dropout = nn.Dropout2d(p=dropout_prob)

        # PReLU layer to apply after concatenating the branches
        self.out_activation = activation()

    def forward(self, x, max_indices, output_size):
        # branch shortcut
        shortcut = self.branch(x)
        shortcut = self.branch_unpool1(
        shortcut, max_indices, output_size=output_size)

        # Main
        main = self.main_conv_1(x)
        main = self.main_transpose_2(main, output_size=output_size)
        main = self.main_bn_2(main)
        main = self.main_act_2(main)
        main = self.main_conv_3(main)
        main = self.dropout(main)

        # Add main and extension branches
        out = main + shortcut
        output = self.out_activation(out)
        return output


class ENet(nn.Module):

    def __init__(self, num_classes, encoder_relu=False, decoder_relu=True):
        super().__init__()

        self.initial_block = InitialBlock(3, 16, relu=encoder_relu)

        # Stage 1 - Encoder
        self.downsample1_0 = DownsamplingBottleneck(
            16,
            64,
            return_indices=True,
            dropout_prob=0.01,
            relu=encoder_relu)
        self.regular1_1 = RegularBottleneck(
            64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_2 = RegularBottleneck(
            64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_3 = RegularBottleneck(
            64, padding=1, dropout_prob=0.01, relu=encoder_relu)
        self.regular1_4 = RegularBottleneck(
            64, padding=1, dropout_prob=0.01, relu=encoder_relu)

        # Stage 2 - Encoder
        self.downsample2_0 = DownsamplingBottleneck(
            64,
            128,
            return_indices=True,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.regular2_1 = RegularBottleneck(
            128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_2 = RegularBottleneck(
            128, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric2_3 = RegularBottleneck(
            128,
            kernel_size=5,
            padding=2,
            asymmetric=True,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.dilated2_4 = RegularBottleneck(
            128, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
        self.regular2_5 = RegularBottleneck(
            128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated2_6 = RegularBottleneck(
            128, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric2_7 = RegularBottleneck(
            128,
            kernel_size=5,
            asymmetric=True,
            padding=2,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.dilated2_8 = RegularBottleneck(
            128, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)

        # Stage 3 - Encoder
        self.regular3_0 = RegularBottleneck(
            128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated3_1 = RegularBottleneck(
            128, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric3_2 = RegularBottleneck(
            128,
            kernel_size=5,
            padding=2,
            asymmetric=True,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.dilated3_3 = RegularBottleneck(
            128, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu)
        self.regular3_4 = RegularBottleneck(
            128, padding=1, dropout_prob=0.1, relu=encoder_relu)
        self.dilated3_5 = RegularBottleneck(
            128, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu)
        self.asymmetric3_6 = RegularBottleneck(
            128,
            kernel_size=5,
            asymmetric=True,
            padding=2,
            dropout_prob=0.1,
            relu=encoder_relu)
        self.dilated3_7 = RegularBottleneck(
            128, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu)

        # Stage 4 - Decoder
        self.upsample4_0 = UpsamplingBottleneck(
            128, 64, dropout_prob=0.1, relu=decoder_relu)
        self.regular4_1 = RegularBottleneck(
            64, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.regular4_2 = RegularBottleneck(
            64, padding=1, dropout_prob=0.1, relu=decoder_relu)

        # Stage 5 - Decoder
        self.upsample5_0 = UpsamplingBottleneck(
            64, 16, dropout_prob=0.1, relu=decoder_relu)
        self.regular5_1 = RegularBottleneck(
            16, padding=1, dropout_prob=0.1, relu=decoder_relu)
        self.transposed_conv = nn.ConvTranspose2d(
            16,
            num_classes,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False)

    def forward(self, x):
        # Initial block
        input_size = x.size()
        x = self.initial_block(x)

        # Stage 1 - Encoder
        stage1_input_size = x.size()
        x, max_indices1_0 = self.downsample1_0(x)
        x = self.regular1_1(x)
        x = self.regular1_2(x)
        x = self.regular1_3(x)
        x = self.regular1_4(x)

        # Stage 2 - Encoder
        stage2_input_size = x.size()
        x, max_indices2_0 = self.downsample2_0(x)
        x = self.regular2_1(x)
        x = self.dilated2_2(x)
        x = self.asymmetric2_3(x)
        x = self.dilated2_4(x)
        x = self.regular2_5(x)
        x = self.dilated2_6(x)
        x = self.asymmetric2_7(x)
        x = self.dilated2_8(x)

        # Stage 3 - Encoder
        x = self.regular3_0(x)
        x = self.dilated3_1(x)
        x = self.asymmetric3_2(x)
        x = self.dilated3_3(x)
        x = self.regular3_4(x)
        x = self.dilated3_5(x)
        x = self.asymmetric3_6(x)
        x = self.dilated3_7(x)

        # Stage 4 - Decoder
        x = self.upsample4_0(x, max_indices2_0, output_size=stage2_input_size)
        x = self.regular4_1(x)
        x = self.regular4_2(x)

        # Stage 5 - Decoder
        x = self.upsample5_0(x, max_indices1_0, output_size=stage1_input_size)
        x = self.regular5_1(x)
        x = self.transposed_conv(x, output_size=input_size)

        return x
