"""修复MLP和UNet模型的兼容性问题"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import sys

# 添加模型路径
models_dir = Path(__file__).parent / 'models'
sys.path.insert(0, str(models_dir))

try:
    from enhanced_mlp import EnhancedMLP1d, EnhancedMLP2d
    from enhanced_unet import EnhancedUNet1d, EnhancedUNet2d
    ENHANCED_MODELS_AVAILABLE = True
    print("✅ 增强模型导入成功")
except ImportError as e:
    print(f"⚠️ 增强模型导入失败: {e}")
    ENHANCED_MODELS_AVAILABLE = False

class CompatibleEnhancedMLP(nn.Module):
    """兼容的增强MLP模型，处理扁平化输入"""
    
    def __init__(self, input_dim, output_dim, hidden_dim=256, num_layers=8, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        
        # 计算输入和输出的空间维度
        self.input_spatial_dim = int(np.sqrt(input_dim))
        self.output_spatial_dim = int(np.sqrt(output_dim))
        
        # 验证是否为完全平方数
        if self.input_spatial_dim ** 2 != input_dim:
            raise ValueError(f"输入维度 {input_dim} 不是完全平方数")
        if self.output_spatial_dim ** 2 != output_dim:
            raise ValueError(f"输出维度 {output_dim} 不是完全平方数")
        
        # 创建网络层
        layers = []
        current_dim = input_dim
        
        # 输入层
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))
        
        # 隐藏层
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        
        # 输出层
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: [batch, input_dim] 扁平化输入
        Returns:
            [batch, output_dim] 扁平化输出
        """
        if len(x.shape) != 2:
            raise ValueError(f"期望输入形状为 [batch, {self.input_dim}], 得到 {x.shape}")
        
        if x.shape[1] != self.input_dim:
            raise ValueError(f"期望输入维度为 {self.input_dim}, 得到 {x.shape[1]}")
        
        output = self.network(x)
        return output

class CompatibleEnhancedUNet(nn.Module):
    """兼容的增强UNet模型，处理扁平化输入"""
    
    def __init__(self, input_dim, output_dim, base_ch=32, num_levels=4, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.base_ch = base_ch
        
        # 计算输入和输出的空间维度
        self.input_spatial_dim = int(np.sqrt(input_dim))
        self.output_spatial_dim = int(np.sqrt(output_dim))
        
        # 验证是否为完全平方数
        if self.input_spatial_dim ** 2 != input_dim:
            raise ValueError(f"输入维度 {input_dim} 不是完全平方数")
        if self.output_spatial_dim ** 2 != output_dim:
            raise ValueError(f"输出维度 {output_dim} 不是完全平方数")
        
        # 输入投影层
        self.input_proj = nn.Linear(input_dim, self.input_spatial_dim * self.input_spatial_dim)
        
        # 编码器
        self.encoder_layers = nn.ModuleList()
        in_ch = 1
        for i in range(num_levels):
            out_ch = base_ch * (2 ** i)
            self.encoder_layers.append(nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ))
            in_ch = out_ch
        
        # 瓶颈层
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_ch, in_ch * 2, 3, padding=1),
            nn.BatchNorm2d(in_ch * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch * 2, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True)
        )
        
        # 解码器
        self.decoder_layers = nn.ModuleList()
        final_out_ch = 1
        for i in range(num_levels - 1, -1, -1):
            out_ch = base_ch * (2 ** i) if i > 0 else final_out_ch
            self.decoder_layers.append(nn.Sequential(
                nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ))
            in_ch = out_ch
        
        # 记录最终输出通道数
        self.final_channels = final_out_ch
        
        # 输出投影层 - 修复维度计算
        # 注意：这里不需要投影层，直接通过插值调整空间维度
        
        # 池化和上采样
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        """
        前向传播
        Args:
            x: [batch, input_dim] 扁平化输入
        Returns:
            [batch, output_dim] 扁平化输出
        """
        if len(x.shape) != 2:
            raise ValueError(f"期望输入形状为 [batch, {self.input_dim}], 得到 {x.shape}")
        
        if x.shape[1] != self.input_dim:
            raise ValueError(f"期望输入维度为 {self.input_dim}, 得到 {x.shape[1]}")
        
        batch_size = x.shape[0]
        
        # 投影到空间维度
        x = self.input_proj(x)  # [batch, spatial_dim^2]
        x = x.view(batch_size, 1, self.input_spatial_dim, self.input_spatial_dim)  # [batch, 1, H, W]
        
        # 编码器路径
        skip_connections = []
        for encoder in self.encoder_layers:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # 瓶颈层
        x = self.bottleneck(x)
        
        # 解码器路径
        for i, decoder in enumerate(self.decoder_layers):
            x = decoder(x)
            if i < len(skip_connections):
                skip = skip_connections[-(i+1)]
                # 确保尺寸匹配
                if x.shape[-2:] != skip.shape[-2:]:
                    x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
                x = x + skip
        
        # 调整到输出空间维度
        if x.shape[-2:] != (self.output_spatial_dim, self.output_spatial_dim):
            x = F.interpolate(x, size=(self.output_spatial_dim, self.output_spatial_dim), 
                            mode='bilinear', align_corners=False)
        
        # 扁平化到输出维度，确保只取单通道
        x = x[:, 0, :, :]  # 取第一个通道 [batch, H, W]
        x = x.view(batch_size, -1)  # [batch, output_dim]
        
        # 确保输出维度正确
        if x.shape[1] != self.output_dim:
            # 如果维度不匹配，使用线性层调整
            if not hasattr(self, 'output_adjust'):
                self.output_adjust = nn.Linear(x.shape[1], self.output_dim).to(x.device)
            x = self.output_adjust(x)
        
        return x

def create_compatible_mlp(input_dim, output_dim, **kwargs):
    """创建兼容的MLP模型"""
    return CompatibleEnhancedMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=kwargs.get('hidden_dims', [256])[0] if kwargs.get('hidden_dims') else 256,
        num_layers=len(kwargs.get('hidden_dims', [256])) + 2,
        dropout=kwargs.get('dropout', 0.1)
    )

def create_compatible_unet(input_dim, output_dim, **kwargs):
    """创建兼容的UNet模型"""
    return CompatibleEnhancedUNet(
        input_dim=input_dim,
        output_dim=output_dim,
        base_ch=kwargs.get('base_ch', 32),
        num_levels=4,
        dropout=kwargs.get('dropout', 0.1)
    )

def test_compatible_models():
    """测试兼容模型"""
    print("🧪 测试兼容模型...")
    
    # 测试数据维度
    input_dim = 400  # 20x20
    output_dim = 40000  # 200x200
    batch_size = 4
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 生成测试数据
    x = torch.randn(batch_size, input_dim).to(device)
    print(f"测试输入形状: {x.shape}")
    
    # 测试MLP
    print("\n📊 测试兼容MLP模型...")
    try:
        mlp_model = create_compatible_mlp(input_dim, output_dim).to(device)
        mlp_output = mlp_model(x)
        print(f"✅ MLP输出形状: {mlp_output.shape}")
        print(f"✅ MLP参数数量: {sum(p.numel() for p in mlp_model.parameters()):,}")
    except Exception as e:
        print(f"❌ MLP测试失败: {e}")
    
    # 测试UNet
    print("\n🏗️ 测试兼容UNet模型...")
    try:
        unet_model = create_compatible_unet(input_dim, output_dim).to(device)
        unet_output = unet_model(x)
        print(f"✅ UNet输出形状: {unet_output.shape}")
        print(f"✅ UNet参数数量: {sum(p.numel() for p in unet_model.parameters()):,}")
    except Exception as e:
        print(f"❌ UNet测试失败: {e}")
    
    print("\n🎉 兼容性测试完成！")

if __name__ == "__main__":
    test_compatible_models()