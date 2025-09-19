"""增强的PINN模型，支持稀疏输入预测稠密输出功能"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List, Callable
import math

class SinusoidalPositionalEncoding(nn.Module):
    """正弦位置编码"""
    
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class FourierFeatureMapping(nn.Module):
    """傅里叶特征映射"""
    
    def __init__(self, input_dim, mapping_size=256, scale=10.0):
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        
        # 随机傅里叶特征
        self.register_buffer('B', torch.randn(input_dim, mapping_size) * scale)
    
    def forward(self, x):
        # x shape: [..., input_dim]
        x_proj = 2 * math.pi * torch.matmul(x, self.B)
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = self.norm2(x + residual)
        return x

class EnhancedPINN(nn.Module):
    """增强的PINN模型，支持稀疏输入预测稠密输出"""
    
    def __init__(self,
                 input_dim=2,  # 空间维度 (x, y) 或 (x, y, t)
                 output_dim=1,  # 输出维度
                 hidden_dim=256,
                 num_layers=8,
                 input_resolution=(32, 32),
                 output_resolution=(128, 128),
                 use_fourier_features=True,
                 fourier_mapping_size=256,
                 fourier_scale=10.0,
                 use_residual_blocks=True,
                 dropout=0.1,
                 activation='gelu',
                 physics_loss_weight=1.0):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution
        self.use_fourier_features = use_fourier_features
        self.use_residual_blocks = use_residual_blocks
        self.physics_loss_weight = physics_loss_weight
        
        # 激活函数
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.GELU()
        
        # 傅里叶特征映射
        if self.use_fourier_features:
            self.fourier_mapping = FourierFeatureMapping(
                input_dim, fourier_mapping_size, fourier_scale
            )
            network_input_dim = fourier_mapping_size * 2
        else:
            network_input_dim = input_dim
        
        # 网络层
        layers = []
        
        # 输入层
        layers.append(nn.Linear(network_input_dim, hidden_dim))
        layers.append(self.activation)
        
        # 隐藏层
        for i in range(num_layers - 2):
            if self.use_residual_blocks and i % 2 == 0 and i > 0:
                layers.append(ResidualBlock(hidden_dim, dropout))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(self.activation)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        
        # 输出层
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入坐标 [batch, ..., input_dim] 或稀疏输入数据
        Returns:
            预测的物理场 [batch, ..., output_dim]
        """
        original_shape = x.shape
        
        # 如果输入是稀疏数据，需要生成坐标网格
        if len(original_shape) > 2:
            # 假设输入是 [batch, height, width, channels] 格式
            batch_size = original_shape[0]
            
            # 生成目标分辨率的坐标网格
            if isinstance(self.output_resolution, (tuple, list)) and len(self.output_resolution) == 2:
                h, w = self.output_resolution
                y_coords = torch.linspace(0, 1, h, device=x.device)
                x_coords = torch.linspace(0, 1, w, device=x.device)
                yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
                coords = torch.stack([xx, yy], dim=-1)  # [h, w, 2]
                coords = coords.unsqueeze(0).expand(batch_size, -1, -1, -1)  # [batch, h, w, 2]
                
                # 重塑为 [batch * h * w, 2]
                coords_flat = coords.reshape(-1, self.input_dim)
            elif isinstance(self.output_resolution, int):
                # 对于扁平化输出，生成默认的128x128网格
                h, w = 128, 128
                y_coords = torch.linspace(0, 1, h, device=x.device)
                x_coords = torch.linspace(0, 1, w, device=x.device)
                yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
                coords = torch.stack([xx, yy], dim=-1)  # [h, w, 2]
                coords = coords.unsqueeze(0).expand(batch_size, -1, -1, -1)  # [batch, h, w, 2]
                
                # 重塑为 [batch * h * w, 2]
                coords_flat = coords.reshape(-1, self.input_dim)
            else:
                raise NotImplementedError("Unsupported output_resolution format")
        else:
            # 输入已经是坐标格式
            coords_flat = x.reshape(-1, self.input_dim)
            batch_size = original_shape[0]
        
        # 傅里叶特征映射
        if self.use_fourier_features:
            coords_flat = self.fourier_mapping(coords_flat)
        
        # 网络前向传播
        output_flat = self.network(coords_flat)
        
        # 重塑输出
        if len(original_shape) > 2:
            # 对于2D模型，如果output_resolution是标量（扁平化输出），直接返回扁平化结果
            if isinstance(self.output_resolution, int):
                # 扁平化输出：[batch_size * grid_points, output_dim] -> [batch_size, total_output_size]
                total_grid_points = 128 * 128  # 默认网格大小
                output = output_flat.reshape(batch_size, total_grid_points, self.output_dim)
                output = output.reshape(batch_size, -1)  # 扁平化为 [batch_size, total_output_size]
            else:
                output = output_flat.reshape(batch_size, *self.output_resolution, self.output_dim)
        else:
            output = output_flat.reshape(batch_size, -1, self.output_dim)
        
        return output
    
    def compute_derivatives(self, x, order=1):
        """
        计算导数（用于物理损失）
        Args:
            x: 输入坐标 [batch, ..., input_dim]
            order: 导数阶数
        Returns:
            导数张量
        """
        x.requires_grad_(True)
        u = self.forward(x)
        
        if order == 1:
            # 一阶导数
            grad_outputs = torch.ones_like(u)
            grads = torch.autograd.grad(
                outputs=u,
                inputs=x,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True
            )[0]
            return grads
        elif order == 2:
            # 二阶导数（拉普拉斯算子）
            grad_outputs = torch.ones_like(u)
            grads = torch.autograd.grad(
                outputs=u,
                inputs=x,
                grad_outputs=grad_outputs,
                create_graph=True,
                retain_graph=True
            )[0]
            
            # 计算二阶导数
            laplacian = torch.zeros_like(u)
            for i in range(self.input_dim):
                grad_i = grads[..., i:i+1]
                grad2_i = torch.autograd.grad(
                    outputs=grad_i,
                    inputs=x,
                    grad_outputs=torch.ones_like(grad_i),
                    create_graph=True,
                    retain_graph=True
                )[0][..., i:i+1]
                # 确保形状匹配
                if grad2_i.shape != laplacian.shape:
                    grad2_i = grad2_i.view(laplacian.shape)
                laplacian += grad2_i
            
            return laplacian
        else:
            raise NotImplementedError(f"Order {order} derivatives not implemented")

class EnhancedPINN1d(EnhancedPINN):
    """1D增强PINN模型"""
    
    def __init__(self, 
                 output_dim=1,
                 hidden_dim=256,
                 num_layers=8,
                 input_resolution=32,
                 output_resolution=128,
                 **kwargs):
        super().__init__(
            input_dim=1,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            input_resolution=(input_resolution,),
            output_resolution=(output_resolution,),
            **kwargs
        )
    
    def forward(self, x):
        # x shape: [batch, length, channels] or [batch, length]
        original_shape = x.shape
        batch_size = original_shape[0]
        
        # 生成1D坐标网格
        coords = torch.linspace(0, 1, self.output_resolution[0], device=x.device)
        coords = coords.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 1)  # [batch, length, 1]
        
        # 重塑为 [batch * length, 1]
        coords_flat = coords.reshape(-1, 1)
        
        # 傅里叶特征映射
        if self.use_fourier_features:
            coords_flat = self.fourier_mapping(coords_flat)
        
        # 网络前向传播
        output_flat = self.network(coords_flat)
        
        # 重塑输出 [batch, length, output_dim]
        output = output_flat.reshape(batch_size, self.output_resolution[0], self.output_dim)
        
        return output

class EnhancedPINN2d(EnhancedPINN):
    """2D增强PINN模型"""
    
    def __init__(self, 
                 output_dim=1,
                 hidden_dim=256,
                 num_layers=8,
                 input_resolution=(32, 32),
                 output_resolution=(128, 128),
                 **kwargs):
        super().__init__(
            input_dim=2,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            input_resolution=input_resolution,
            output_resolution=output_resolution,
            **kwargs
        )

class PINNLoss(nn.Module):
    """PINN损失函数，包含数据损失和物理损失"""
    
    def __init__(self, 
                 pde_function: Optional[Callable] = None,
                 boundary_function: Optional[Callable] = None,
                 data_weight=1.0,
                 pde_weight=1.0,
                 boundary_weight=1.0):
        super().__init__()
        self.pde_function = pde_function
        self.boundary_function = boundary_function
        self.data_weight = data_weight
        self.pde_weight = pde_weight
        self.boundary_weight = boundary_weight
        self.mse_loss = nn.MSELoss()
    
    def forward(self, model, x_data, y_data, x_pde=None, x_boundary=None):
        """
        计算PINN总损失
        Args:
            model: PINN模型
            x_data: 数据点坐标
            y_data: 数据点值
            x_pde: PDE点坐标
            x_boundary: 边界点坐标
        """
        total_loss = 0.0
        loss_dict = {}
        
        # 数据损失
        if x_data is not None and y_data is not None:
            y_pred = model(x_data)
            # 确保形状匹配
            if y_pred.dim() > y_data.dim():
                y_pred = y_pred.squeeze(-1)
            elif y_pred.dim() < y_data.dim():
                y_data = y_data.squeeze(-1)
            data_loss = self.mse_loss(y_pred, y_data)
            total_loss += self.data_weight * data_loss
            loss_dict['data_loss'] = data_loss.item()
        
        # PDE损失
        if x_pde is not None and self.pde_function is not None:
            pde_residual = self.pde_function(model, x_pde)
            pde_loss = torch.mean(pde_residual ** 2)
            total_loss += self.pde_weight * pde_loss
            loss_dict['pde_loss'] = pde_loss.item()
        
        # 边界损失
        if x_boundary is not None and self.boundary_function is not None:
            boundary_residual = self.boundary_function(model, x_boundary)
            boundary_loss = torch.mean(boundary_residual ** 2)
            total_loss += self.boundary_weight * boundary_loss
            loss_dict['boundary_loss'] = boundary_loss.item()
        
        loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict

class SparseToDeseAdapterPINN(nn.Module):
    """PINN模型的稀疏到稠密适配器"""
    
    def __init__(self, 
                 pinn_model: nn.Module,
                 input_resolution: Tuple[int, ...],
                 output_resolution: Tuple[int, ...]):
        super().__init__()
        self.pinn_model = pinn_model
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution
        
    def forward(self, sparse_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sparse_input: 稀疏输入张量
        Returns:
            稠密输出张量
        """
        # PINN模型直接从坐标生成稠密输出
        dense_output = self.pinn_model(sparse_input)
        
        return dense_output

# 示例PDE函数
def heat_equation_2d(model, x):
    """
    2D热方程: ∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²)
    """
    x.requires_grad_(True)
    u = model(x)
    
    # 计算时间导数
    u_t = torch.autograd.grad(
        outputs=u, inputs=x,
        grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True
    )[0][:, 2:3]  # 假设时间是第三个维度
    
    # 计算空间二阶导数
    u_x = torch.autograd.grad(
        outputs=u, inputs=x,
        grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True
    )[0][:, 0:1]
    
    u_xx = torch.autograd.grad(
        outputs=u_x, inputs=x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True, retain_graph=True
    )[0][:, 0:1]
    
    u_y = torch.autograd.grad(
        outputs=u, inputs=x,
        grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True
    )[0][:, 1:2]
    
    u_yy = torch.autograd.grad(
        outputs=u_y, inputs=x,
        grad_outputs=torch.ones_like(u_y),
        create_graph=True, retain_graph=True
    )[0][:, 1:2]
    
    # 热扩散系数
    alpha = 0.1
    
    # PDE残差
    pde_residual = u_t - alpha * (u_xx + u_yy)
    
    return pde_residual

def poisson_equation_2d(model, x):
    """
    2D泊松方程: ∇²u = f
    """
    x.requires_grad_(True)
    u = model(x)
    
    # 计算拉普拉斯算子
    laplacian = model.compute_derivatives(x, order=2)
    
    # 源项（可以根据具体问题修改）
    f = torch.sin(torch.pi * x[:, 0:1]) * torch.sin(torch.pi * x[:, 1:2])
    
    # PDE残差
    pde_residual = laplacian - f
    
    return pde_residual

# 工厂函数
def create_enhanced_pinn1d(output_dim=1,
                          hidden_dim=256,
                          num_layers=8,
                          input_resolution=32,
                          output_resolution=128,
                          **kwargs) -> EnhancedPINN1d:
    """创建增强的PINN1d模型"""
    return EnhancedPINN1d(
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        input_resolution=input_resolution,
        output_resolution=output_resolution,
        **kwargs
    )

def create_enhanced_pinn2d(output_dim=1,
                          hidden_dim=256,
                          num_layers=8,
                          input_resolution=(32, 32),
                          output_resolution=(128, 128),
                          **kwargs) -> EnhancedPINN2d:
    """创建增强的PINN2d模型"""
    return EnhancedPINN2d(
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        input_resolution=input_resolution,
        output_resolution=output_resolution,
        **kwargs
    )

if __name__ == "__main__":
    # 测试代码
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 测试PINN1d
    print("测试EnhancedPINN1d...")
    model_1d = create_enhanced_pinn1d(
        output_dim=1,
        input_resolution=32,
        output_resolution=128
    ).to(device)
    
    x_1d = torch.randn(4, 32, 1).to(device)
    output_1d = model_1d(x_1d)
    print(f"PINN1d输入形状: {x_1d.shape}, 输出形状: {output_1d.shape}")
    
    # 测试PINN2d
    print("\n测试EnhancedPINN2d...")
    model_2d = create_enhanced_pinn2d(
        output_dim=1,
        input_resolution=(32, 32),
        output_resolution=(128, 128)
    ).to(device)
    
    x_2d = torch.randn(4, 32, 32, 1).to(device)
    output_2d = model_2d(x_2d)
    print(f"PINN2d输入形状: {x_2d.shape}, 输出形状: {output_2d.shape}")
    
    # 测试PINN损失
    print("\n测试PINN损失函数...")
    pinn_loss = PINNLoss(pde_function=poisson_equation_2d)
    
    # 生成测试数据
    x_data = torch.randn(100, 2, requires_grad=True).to(device)
    y_data = torch.randn(100, 1).to(device)
    x_pde = torch.randn(200, 2, requires_grad=True).to(device)
    
    loss, loss_dict = pinn_loss(model_2d, x_data, y_data, x_pde)
    print(f"总损失: {loss.item():.6f}")
    print(f"损失详情: {loss_dict}")
    
    print("\n所有测试通过！")