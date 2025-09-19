"""增强的UNet模型，支持稀疏输入预测稠密输出功能"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)

class EnhancedUNet1d(nn.Module):
    """增强的UNet1d模型，支持稀疏输入预测稠密输出"""
    
    def __init__(self, 
                 in_channels=1, 
                 out_channels=1, 
                 init_features=32,
                 input_resolution=32,
                 output_resolution=128,
                 use_upsampling=True):
        super().__init__()
        
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution
        self.use_upsampling = use_upsampling
        
        features = init_features
        
        # Encoder
        self.encoder1 = self._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder2 = self._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder3 = self._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.encoder4 = self._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = self._block(features * 8, features * 16, name="bottleneck")
        
        # Decoder
        self.upconv4 = nn.ConvTranspose1d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = self._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose1d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose1d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose1d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features, name="dec1")
        
        # 分辨率调整层
        if self.use_upsampling and output_resolution > input_resolution:
            scale_factor = output_resolution / input_resolution
            if scale_factor == 4:
                self.resolution_adapter = nn.Sequential(
                    nn.ConvTranspose1d(features, features, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose1d(features, features, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(inplace=True)
                )
            elif scale_factor == 2:
                self.resolution_adapter = nn.Sequential(
                    nn.ConvTranspose1d(features, features, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(inplace=True)
                )
            else:
                self.resolution_adapter = nn.Identity()
        else:
            self.resolution_adapter = nn.Identity()
        
        # 输出层
        self.conv = nn.Conv1d(features, out_channels, kernel_size=1)
        
        # 输出投影层，用于调整输出长度
        self.output_projection = nn.Linear(output_resolution, output_resolution)
    
    def _block(self, in_channels, features, name):
        return nn.Sequential(
            nn.Conv1d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm1d(features),  # 使用InstanceNorm1d替代BatchNorm1d
            nn.ReLU(inplace=True),
            nn.Conv1d(features, features, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm1d(features),  # 使用InstanceNorm1d替代BatchNorm1d
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        # 处理不同的输入格式
        if x.dim() == 2:
            # [batch, length] -> [batch, 1, length]
            x = x.unsqueeze(1)
        elif x.dim() == 3:
            if x.shape[-1] < x.shape[1]:
                # [batch, length, channels] -> [batch, channels, length]
                x = x.permute(0, 2, 1)
            # 如果通道数不匹配，调整到期望的通道数
            if x.shape[1] != self.encoder1[0].in_channels:
                # 使用1x1卷积调整通道数
                channel_adapter = nn.Conv1d(x.shape[1], self.encoder1[0].in_channels, kernel_size=1).to(x.device)
                x = channel_adapter(x)
        
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        # 处理skip connection，调整尺寸匹配
        if enc1.shape[-1] != dec1.shape[-1]:
            enc1 = F.interpolate(enc1, size=dec1.shape[-1], mode='linear', align_corners=False)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        # 分辨率调整
        if self.use_upsampling and self.output_resolution > self.input_resolution:
            dec1 = self.resolution_adapter(dec1)
            # 精确调整到目标分辨率
            if dec1.shape[-1] != self.output_resolution:
                dec1 = F.interpolate(dec1, size=self.output_resolution, mode='linear', align_corners=False)
        elif self.output_resolution != self.input_resolution:
            dec1 = F.interpolate(dec1, size=self.output_resolution, mode='linear', align_corners=False)
        
        # 输出
        output = self.conv(dec1)
        
        # 确保输出长度匹配期望
        if output.shape[-1] != self.output_resolution:
            output = F.interpolate(output, size=self.output_resolution, mode='linear', align_corners=False)
        
        # 转换为扁平化输出格式 [batch, total_length]
        batch_size = output.shape[0]
        output = output.view(batch_size, -1)
        
        return output

class EnhancedUNet2d(nn.Module):
    """增强的UNet2d模型，支持稀疏输入预测稠密输出"""
    
    def __init__(self, 
                 in_channels=1, 
                 out_channels=1, 
                 init_features=32,
                 input_resolution=(32, 32),
                 output_resolution=(128, 128),
                 use_upsampling=True,
                 bilinear=False):
        super().__init__()
        
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution
        self.use_upsampling = use_upsampling
        self.bilinear = bilinear
        
        features = init_features
        
        # 输入层 - 确保正确处理输入通道数
        self.inc = DoubleConv(in_channels, features)
        
        # Encoder
        self.down1 = Down(features, features * 2)
        self.down2 = Down(features * 2, features * 4)
        self.down3 = Down(features * 4, features * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(features * 8, features * 16 // factor)
        
        # Decoder
        self.up1 = Up(features * 16, features * 8 // factor, bilinear)
        self.up2 = Up(features * 8, features * 4 // factor, bilinear)
        self.up3 = Up(features * 4, features * 2 // factor, bilinear)
        self.up4 = Up(features * 2, features, bilinear)
        
        # 分辨率调整层
        if self.use_upsampling and (output_resolution[0] > input_resolution[0] or 
                                   output_resolution[1] > input_resolution[1]):
            scale_h = output_resolution[0] / input_resolution[0]
            scale_w = output_resolution[1] / input_resolution[1]
            
            if scale_h == 4 and scale_w == 4:
                self.resolution_adapter = nn.Sequential(
                    nn.ConvTranspose2d(features, features, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(features, features, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(inplace=True)
                )
            elif scale_h == 2 and scale_w == 2:
                self.resolution_adapter = nn.Sequential(
                    nn.ConvTranspose2d(features, features, kernel_size=4, stride=2, padding=1),
                    nn.ReLU(inplace=True)
                )
            else:
                self.resolution_adapter = nn.Identity()
        else:
            self.resolution_adapter = nn.Identity()
        
        # 输出层
        self.outc = OutConv(features, out_channels)
    
    def forward(self, x):
        # 处理不同的输入格式
        original_shape = x.shape
        
        if x.dim() == 2:
            # [batch, flattened] -> [batch, channels, height, width]
            batch_size = x.shape[0]
            # 假设输入是方形的，计算边长
            spatial_size = int(np.sqrt(x.shape[1]))
            if spatial_size * spatial_size != x.shape[1]:
                # 如果不是完全平方数，调整到最接近的平方数
                spatial_size = int(np.sqrt(x.shape[1])) + 1
                x = F.pad(x, (0, spatial_size * spatial_size - x.shape[1]))
            x = x.view(batch_size, 1, spatial_size, spatial_size)
        elif x.dim() == 3:
            # [batch, height, width] -> [batch, 1, height, width]
            x = x.unsqueeze(1)
        elif x.dim() == 4:
            if x.shape[1] > x.shape[-1] and x.shape[-1] <= 16:
                # [batch, height, width, channels] -> [batch, channels, height, width]
                x = x.permute(0, 3, 1, 2)
        
        # 确保输入尺寸足够大，避免下采样过程中尺寸变为0
        if x.shape[-1] < 16 or x.shape[-2] < 16:
            # 如果输入太小，先上采样到合适尺寸
            target_size = max(16, max(x.shape[-2], x.shape[-1]))
            x = F.interpolate(x, size=(target_size, target_size), mode='bilinear', align_corners=False)
        
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # 分辨率调整 - 修复输出尺寸计算
        current_size = (x.shape[-2], x.shape[-1])
        if self.use_upsampling and (self.output_resolution[0] > self.input_resolution[0] or 
                                   self.output_resolution[1] > self.input_resolution[1]):
            x = self.resolution_adapter(x)
            current_size = (x.shape[-2], x.shape[-1])
        
        # 始终确保输出尺寸正确
        if current_size != self.output_resolution:
            x = F.interpolate(x, size=self.output_resolution, mode='bilinear', align_corners=False)
        
        # 输出
        output = self.outc(x)
        
        # 根据期望的输出格式进行转换
        if len(original_shape) == 2:
            # 如果输入是扁平化的，输出也应该是扁平化的
            batch_size = output.shape[0]
            output = output.view(batch_size, -1)
            # 确保输出长度匹配期望
            target_length = self.output_resolution[0] * self.output_resolution[1]
            if output.shape[1] != target_length:
                # 使用线性插值调整长度
                output = F.interpolate(output.unsqueeze(1), size=target_length, mode='linear', align_corners=False).squeeze(1)
        else:
            # 转换回 [batch, height, width, channels] 格式
            output = output.permute(0, 2, 3, 1)
        
        return output

class EnhancedUNet3d(nn.Module):
    """增强的UNet3d模型，支持稀疏输入预测稠密输出"""
    
    def __init__(self, 
                 in_channels=1, 
                 out_channels=1, 
                 init_features=32,
                 input_resolution=(16, 32, 32),
                 output_resolution=(32, 128, 128),
                 use_upsampling=True):
        super().__init__()
        
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution
        self.use_upsampling = use_upsampling
        
        features = init_features
        
        # Encoder
        self.encoder1 = self._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder2 = self._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.encoder3 = self._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = self._block(features * 4, features * 8, name="bottleneck")
        
        # Decoder
        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features, name="dec1")
        
        # 分辨率调整层
        if self.use_upsampling:
            self.resolution_adapter = nn.Sequential(
                nn.ConvTranspose3d(features, features, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            self.resolution_adapter = nn.Identity()
        
        # 输出层
        self.conv = nn.Conv3d(features, out_channels, kernel_size=1)
    
    def _block(self, in_channels, features, name):
        return nn.Sequential(
            nn.Conv3d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True),
            nn.Conv3d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        # x shape: [batch, depth, height, width, channels] or [batch, channels, depth, height, width]
        if x.dim() == 5 and x.shape[1] > x.shape[-1]:
            # 如果第二维大于最后一维，假设是 [batch, depth, height, width, channels]
            x = x.permute(0, 4, 1, 2, 3)  # 转换为 [batch, channels, depth, height, width]
        
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool3(enc3))
        
        # Decoder
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        # 分辨率调整
        if self.use_upsampling:
            dec1 = self.resolution_adapter(dec1)
            # 精确调整到目标分辨率
            if dec1.shape[-3:] != self.output_resolution:
                dec1 = F.interpolate(dec1, size=self.output_resolution, mode='trilinear', align_corners=False)
        elif self.output_resolution != self.input_resolution:
            dec1 = F.interpolate(dec1, size=self.output_resolution, mode='trilinear', align_corners=False)
        
        # 输出
        output = self.conv(dec1)
        
        # 转换回 [batch, depth, height, width, channels] 格式
        output = output.permute(0, 2, 3, 4, 1)
        
        return output

class SparseToDeseAdapterUNet(nn.Module):
    """UNet模型的稀疏到稠密适配器"""
    
    def __init__(self, 
                 unet_model: nn.Module,
                 input_resolution: Tuple[int, ...],
                 output_resolution: Tuple[int, ...]):
        super().__init__()
        self.unet_model = unet_model
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution
        
    def forward(self, sparse_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sparse_input: 稀疏输入张量
        Returns:
            稠密输出张量
        """
        # 调用UNet模型
        dense_output = self.unet_model(sparse_input)
        
        return dense_output

# 工厂函数
def create_enhanced_unet1d(in_channels=1, 
                          out_channels=1, 
                          init_features=32,
                          input_resolution=32,
                          output_resolution=128,
                          use_upsampling=True) -> EnhancedUNet1d:
    """创建增强的UNet1d模型"""
    return EnhancedUNet1d(
        in_channels=in_channels,
        out_channels=out_channels,
        init_features=init_features,
        input_resolution=input_resolution,
        output_resolution=output_resolution,
        use_upsampling=use_upsampling
    )

def create_enhanced_unet2d(in_channels=1, 
                          out_channels=1, 
                          init_features=32,
                          input_resolution=(32, 32),
                          output_resolution=(128, 128),
                          use_upsampling=True,
                          bilinear=False) -> EnhancedUNet2d:
    """创建增强的UNet2d模型"""
    return EnhancedUNet2d(
        in_channels=in_channels,
        out_channels=out_channels,
        init_features=init_features,
        input_resolution=input_resolution,
        output_resolution=output_resolution,
        use_upsampling=use_upsampling,
        bilinear=bilinear
    )

def create_enhanced_unet3d(in_channels=1, 
                          out_channels=1, 
                          init_features=32,
                          input_resolution=(16, 32, 32),
                          output_resolution=(32, 128, 128),
                          use_upsampling=True) -> EnhancedUNet3d:
    """创建增强的UNet3d模型"""
    return EnhancedUNet3d(
        in_channels=in_channels,
        out_channels=out_channels,
        init_features=init_features,
        input_resolution=input_resolution,
        output_resolution=output_resolution,
        use_upsampling=use_upsampling
    )

if __name__ == "__main__":
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试UNet1d
    print("测试EnhancedUNet1d...")
    model_1d = create_enhanced_unet1d(
        in_channels=1,
        out_channels=1,
        input_resolution=32,
        output_resolution=128
    ).to(device)
    
    x_1d = torch.randn(4, 32, 1).to(device)
    output_1d = model_1d(x_1d)
    print(f"UNet1d输入形状: {x_1d.shape}, 输出形状: {output_1d.shape}")
    
    # 测试UNet2d
    print("\n测试EnhancedUNet2d...")
    model_2d = create_enhanced_unet2d(
        in_channels=1,
        out_channels=1,
        input_resolution=(32, 32),
        output_resolution=(128, 128)
    ).to(device)
    
    x_2d = torch.randn(4, 32, 32, 1).to(device)
    output_2d = model_2d(x_2d)
    print(f"UNet2d输入形状: {x_2d.shape}, 输出形状: {output_2d.shape}")
    
    # 测试UNet3d
    print("\n测试EnhancedUNet3d...")
    model_3d = create_enhanced_unet3d(
        in_channels=1,
        out_channels=1,
        input_resolution=(16, 32, 32),
        output_resolution=(32, 128, 128)
    ).to(device)
    
    x_3d = torch.randn(4, 16, 32, 32, 1).to(device)
    output_3d = model_3d(x_3d)
    print(f"UNet3d输入形状: {x_3d.shape}, 输出形状: {output_3d.shape}")
    
    print("\n所有测试通过！")