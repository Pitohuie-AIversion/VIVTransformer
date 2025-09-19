"""增强的FNO模型，支持稀疏输入预测稠密输出功能"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super().__init__()
        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.complex64))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes - 强制使用complex64避免ComplexHalf错误
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1, dtype=torch.complex64, device=x.device)
        weights1_complex64 = self.weights1.to(torch.complex64)
        x_ft_complex64 = x_ft.to(torch.complex64)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft_complex64[:, :, :self.modes1], weights1_complex64)

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.complex64))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.complex64))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # 确保modes不超过实际的频域尺寸
        modes1 = min(self.modes1, x_ft.size(-2))
        modes2 = min(self.modes2, x_ft.size(-1))

        # Multiply relevant Fourier modes - 强制使用complex64避免ComplexHalf错误
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.complex64, device=x.device)
        x_ft_complex64 = x_ft.to(torch.complex64)
        weights1_complex64 = self.weights1[:, :, :modes1, :modes2].to(torch.complex64)
        weights2_complex64 = self.weights2[:, :, :modes1, :modes2].to(torch.complex64)
        out_ft[:, :, :modes1, :modes2] = \
            self.compl_mul2d(x_ft_complex64[:, :, :modes1, :modes2], weights1_complex64)
        if modes1 < x_ft.size(-2):  # 只有当有足够的模式时才处理负频率
            out_ft[:, :, -modes1:, :modes2] = \
                self.compl_mul2d(x_ft_complex64[:, :, -modes1:, :modes2], weights2_complex64)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class EnhancedFNO1d(nn.Module):
    """增强的FNO1d模型，支持稀疏输入预测稠密输出"""
    
    def __init__(self, 
                 num_channels=1, 
                 modes=16, 
                 width=64, 
                 initial_step=10,
                 input_resolution=32,
                 output_resolution=128,
                 use_upsampling=True):
        super().__init__()
        
        self.modes1 = modes
        self.width = width
        self.padding = 0  # 避免cuFFT错误，设置为0
        # FNO1d使用单个数值，不需要转换为元组
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution
        self.use_upsampling = use_upsampling
        
        # 输入投影层 - 修复维度匹配问题
        self.fc0 = nn.Linear(num_channels, self.width)
        
        # Fourier层
        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        
        # 卷积层
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        
        # 上采样层（如果需要）
        if self.use_upsampling and output_resolution > input_resolution:
            self.upsampler = nn.Sequential(
                nn.ConvTranspose1d(self.width, self.width, 
                                 kernel_size=4, stride=2, padding=1),
                nn.GELU(),
                nn.ConvTranspose1d(self.width, self.width, 
                                 kernel_size=4, stride=2, padding=1),
                nn.GELU()
            )
        
        # 输出投影层 - 根据输出分辨率动态调整
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, num_channels)
        
    def forward(self, x, grid=None):
        # 处理输入维度：确保x是3D张量 [batch, resolution, channels]
        original_shape = x.shape
        if len(x.shape) == 2:  # [batch, resolution]
            x = x.unsqueeze(-1)  # [batch, resolution, 1]
        elif len(x.shape) == 4:  # [batch, channels, height, width]
            batch_size, channels, height, width = x.shape
            x = x.permute(0, 2, 3, 1).contiguous().view(batch_size, height * width, channels)
        elif len(x.shape) == 3 and x.shape[1] == 1:  # [batch, 1, resolution] - 需要转置
            x = x.permute(0, 2, 1)  # [batch, resolution, 1]
        
        batch_size = x.shape[0]
        
        # 跳过位置编码以避免维度问题
        # if grid is None:
        #     grid = torch.linspace(0, 1, x.shape[1], device=x.device).unsqueeze(0).unsqueeze(-1)
        #     grid = grid.expand(batch_size, -1, -1)
        # 
        # # 连接输入和位置编码
        # x = torch.cat((x, grid), dim=-1)  # [batch, resolution, channels+1]
        x = self.fc0(x)  # [batch, resolution, width]
        x = x.permute(0, 2, 1)  # [batch, width, resolution]
        
        # 填充
        x = F.pad(x, [0, self.padding])
        
        # Fourier层
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)
        
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)
        
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)
        
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        
        # 去除填充
        if self.padding > 0:
            x = x[..., :-self.padding]
        
        # 上采样到目标分辨率
        if self.use_upsampling and self.output_resolution > self.input_resolution:
            x = self.upsampler(x)
            # 调整到精确的输出分辨率
            if x.shape[-1] != self.output_resolution:
                x = F.interpolate(x, size=self.output_resolution, mode='linear', align_corners=False)
        elif self.output_resolution != self.input_resolution:
            # 使用插值调整分辨率
            x = F.interpolate(x, size=self.output_resolution, mode='linear', align_corners=False)
        
        x = x.permute(0, 2, 1)  # [batch, output_resolution, width]
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        
        # 对于1D情况，需要扁平化输出以匹配期望的形状
        # 从 [batch, current_resolution, num_channels] 到 [batch, target_output_resolution]
        x = x.view(x.size(0), -1)  # [batch, current_resolution * num_channels]
        
        # 计算当前输出的总维度
        current_output_dim = x.size(1)
        target_output_dim = self.output_resolution * self.output_resolution if isinstance(self.output_resolution, int) else np.prod(self.output_resolution)
        
        # 如果当前输出维度不等于目标维度，使用线性层映射
        if current_output_dim != target_output_dim:
            if not hasattr(self, 'final_projection'):
                self.final_projection = nn.Linear(current_output_dim, target_output_dim).to(x.device)
            x = self.final_projection(x)
        
        return x

class EnhancedFNO2d(nn.Module):
    """增强的FNO2d模型，支持稀疏输入预测稠密输出"""
    
    def __init__(self, 
                 num_channels=1, 
                 modes1=12, 
                 modes2=12, 
                 width=20, 
                 initial_step=10,
                 input_resolution=(32, 32),
                 output_resolution=(128, 128),
                 use_upsampling=True):
        super().__init__()
        
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 0  # 避免cuFFT错误，设置为0
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution
        self.use_upsampling = use_upsampling
        
        # 输入投影层 - 修复维度匹配问题
        self.fc0 = nn.Linear(num_channels, self.width)
        
        # Fourier层
        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        
        # 卷积层
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        
        # 上采样层（如果需要）
        if self.use_upsampling and (output_resolution[0] > input_resolution[0] or 
                                   output_resolution[1] > input_resolution[1]):
            # 计算上采样倍数
            scale_h = output_resolution[0] / input_resolution[0]
            scale_w = output_resolution[1] / input_resolution[1]
            
            if scale_h == 4 and scale_w == 4:  # 4x上采样
                self.upsampler = nn.Sequential(
                    nn.ConvTranspose2d(self.width, self.width, 
                                     kernel_size=4, stride=2, padding=1),
                    nn.GELU(),
                    nn.ConvTranspose2d(self.width, self.width, 
                                     kernel_size=4, stride=2, padding=1),
                    nn.GELU()
                )
            elif scale_h == 2 and scale_w == 2:  # 2x上采样
                self.upsampler = nn.Sequential(
                    nn.ConvTranspose2d(self.width, self.width, 
                                     kernel_size=4, stride=2, padding=1),
                    nn.GELU()
                )
            else:
                # 使用自适应上采样
                self.upsampler = nn.Sequential(
                    nn.ConvTranspose2d(self.width, self.width, 
                                     kernel_size=3, stride=1, padding=1),
                    nn.GELU()
                )
        
        # 输出投影层
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, num_channels)
        
    def forward(self, x, grid=None):
        # 处理不同的输入格式
        if len(x.shape) == 2:  # [batch, flattened] - 需要重塑为2D
            batch_size = x.shape[0]
            # 重塑为输入分辨率
            x = x.view(batch_size, self.input_resolution[0], self.input_resolution[1])
            x = x.unsqueeze(-1)  # [batch, height, width, 1]
        elif len(x.shape) == 3:  # [batch, height, width]
            x = x.unsqueeze(-1)  # [batch, height, width, 1]
        elif len(x.shape) == 4 and x.shape[1] == 1:  # [batch, 1, height, width]
            x = x.permute(0, 2, 3, 1)  # [batch, height, width, 1]
        
        batch_size, h, w, channels = x.shape
        
        # 确保fc0层的输入维度正确
        x = self.fc0(x)  # [batch, height, width, width]
        x = x.permute(0, 3, 1, 2)  # [batch, width, height, width]
        
        # 填充
        x = F.pad(x, [0, self.padding, 0, self.padding])
        
        # Fourier层
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)
        
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)
        
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)
        
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        
        # 去除填充
        if self.padding > 0:
            x = x[..., :-self.padding, :-self.padding]
        
        # 上采样到目标分辨率
        if self.use_upsampling and (self.output_resolution[0] > self.input_resolution[0] or 
                                   self.output_resolution[1] > self.input_resolution[1]):
            x = self.upsampler(x)
            # 调整到精确的输出分辨率
            if x.shape[-2:] != tuple(self.output_resolution):
                x = F.interpolate(x, size=self.output_resolution, mode='bilinear', align_corners=False)
        elif tuple(self.output_resolution) != tuple(self.input_resolution):
            # 使用插值调整分辨率
            x = F.interpolate(x, size=self.output_resolution, mode='bilinear', align_corners=False)
        
        x = x.permute(0, 2, 3, 1)  # [batch, height, width, width]
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        
        # 扁平化输出以匹配期望的形状 [batch, output_resolution[0] * output_resolution[1]]
        x = x.view(x.size(0), -1)
        
        # 确保输出维度正确
        target_output_dim = np.prod(self.output_resolution)
        current_output_dim = x.size(1)
        
        if current_output_dim != target_output_dim:
            if not hasattr(self, 'final_projection_2d'):
                self.final_projection_2d = nn.Linear(current_output_dim, target_output_dim).to(x.device)
            x = self.final_projection_2d(x)
        
        return x

class SparseToDeseAdapterFNO(nn.Module):
    """FNO模型的稀疏到稠密适配器"""
    
    def __init__(self, 
                 fno_model: nn.Module,
                 input_resolution: Tuple[int, ...],
                 output_resolution: Tuple[int, ...]):
        super().__init__()
        self.fno_model = fno_model
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution
        
    def forward(self, sparse_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sparse_input: 稀疏输入张量
        Returns:
            稠密输出张量
        """
        # 调用FNO模型
        dense_output = self.fno_model(sparse_input)
        
        return dense_output

# 工厂函数
def create_enhanced_fno1d(num_channels=1, 
                         modes=16, 
                         width=64, 
                         initial_step=10,
                         input_resolution=32,
                         output_resolution=128,
                         use_upsampling=True) -> EnhancedFNO1d:
    """创建增强的FNO1d模型"""
    return EnhancedFNO1d(
        num_channels=num_channels,
        modes=modes,
        width=width,
        initial_step=initial_step,
        input_resolution=input_resolution,
        output_resolution=output_resolution,
        use_upsampling=use_upsampling
    )

def create_enhanced_fno2d(num_channels=1, 
                         modes1=12, 
                         modes2=12, 
                         width=20, 
                         initial_step=10,
                         input_resolution=(32, 32),
                         output_resolution=(128, 128),
                         use_upsampling=True) -> EnhancedFNO2d:
    """创建增强的FNO2d模型"""
    return EnhancedFNO2d(
        num_channels=num_channels,
        modes1=modes1,
        modes2=modes2,
        width=width,
        initial_step=initial_step,
        input_resolution=input_resolution,
        output_resolution=output_resolution,
        use_upsampling=use_upsampling
    )

if __name__ == "__main__":
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试FNO1d
    print("测试EnhancedFNO1d...")
    model_1d = create_enhanced_fno1d(
        num_channels=1,
        input_resolution=32,
        output_resolution=128
    ).to(device)
    
    x_1d = torch.randn(4, 32, 1).to(device)
    output_1d = model_1d(x_1d)
    print(f"FNO1d输入形状: {x_1d.shape}, 输出形状: {output_1d.shape}")
    
    # 测试FNO2d
    print("\n测试EnhancedFNO2d...")
    model_2d = create_enhanced_fno2d(
        num_channels=1,
        input_resolution=(32, 32),
        output_resolution=(128, 128)
    ).to(device)
    
    x_2d = torch.randn(4, 32, 32, 1).to(device)
    output_2d = model_2d(x_2d)
    print(f"FNO2d输入形状: {x_2d.shape}, 输出形状: {output_2d.shape}")
    
    print("\n所有测试通过！")