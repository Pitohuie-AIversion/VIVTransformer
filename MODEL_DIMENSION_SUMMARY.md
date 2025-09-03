# 模型与数据维度总结（Dimension & Shape Summary）

本文件汇总了当前项目中模型、数据集、训练循环与配置在“维度/形状”方面的约定、核查结果与改进建议，便于后续排错与扩展。

## 1. 关键文件与数据流概览
- 模型定义：modify_multi_attention/mymodels/transformer.py 中的 TransformerFlowReconstructionModel，包含初始化与 forward，决定输入嵌入、时间步编码、位置编码（1D 可学习/正弦、2D 可分离可学习）、编码器/解码器以及输出头（global / per_token）。
- 配置来源：modify_multi_attention/configs/config.yaml 与 config_pdebench.yaml 的 model 段，参数在 modify_multi_attention/main.py 中用于实例化模型。
- 数据与加载：modify_multi_attention/utils/dataset.py（PressureDataset）、modify_multi_attention/utils/pdebench_dataset.py（PDEBenchDataset）、modify_multi_attention/utils/dataloader.py（collate_fn、get_loaders）。
- 训练循环：modify_multi_attention/training/trainer.py 的 train_model 中，按批解包 (in_press, out_pressure, time_steps) 并调用 model.forward。

## 2. 模型维度约定与前向路径
- 输入张量：
  - x_in_pressures_flat: 形状 [B, input_dim]，来自数据集将 2D 网格展平成 1D 向量。
  - x_time_steps: 形状 [B] 或 [B, 1]，时间步ID或连续值。
- 序列长度与网格：
  - 若配置提供 input_hw = [H, W]，则 seq_len = H × W；否则使用配置中的 seq_len。
  - 输入展平向量会以 seq_len 为粒度切分/重排，随后映射到 d_model 并 reshape 为 [B, seq_len, d_model]。
- 位置与时间编码：
  - 支持 1D（可学习/正弦）与 2D 可分离可学习位置编码；时间步可用 embedding 或 MLP 编码并与 token 表示相加/拼接。
- 编解码与输出头：
  - 编码器/解码器输出 token 表示后，依据 output_head_type 分两种：
    - global：聚合（或使用特殊token）得到 [B, d_model]，再线性映射到 [B, output_dim]。
    - per_token：对每个 token 做线性映射得到 [B, seq_len, out_channels_per_token]，若最终需要对齐展平目标，可令 output_dim = seq_len × out_channels_per_token 并按需 reshape/flatten。

## 3. 数据集与批数据形状
- PressureDataset：
  - __getitem__ 返回：in_press_flat [input_dim]、pressure_flat [output_dim]、time_step [标量]。
- PDEBenchDataset：
  - 读取 HDF5 后将输入/目标展平成 1D 向量，时间步信息一并返回。
- DataLoader 与 collate_fn：
  - collate_fn 过滤空样本后调用 default_collate，得到批：
    - in_press [B, input_dim]
    - out_pressure [B, output_dim]
    - time_steps [B] 或 [B, 1]

## 4. 训练循环中的形状一致性
- 训练循环解包：for i, (in_press, out_pressure, time_steps) in enumerate(train_loader):
- 前向调用：model(in_press, time_steps) 与损失计算 criterion(pred, out_pressure)；与模型的 forward 约定一致。

## 5. 配置核查与一致性建议
- config.yaml（示例）：
  - model: input_dim: 400, output_dim: 40000, d_model: 512, seq_len: 32, input_hw: [20, 20], num_heads/layers 等。
  - 注意：若提供 input_hw，实际 seq_len 将以 H×W 优先生效（此例为 400），可能和显式设置的 seq_len: 32 不一致。建议：
    - 若希望基于网格建模，保留 input_hw 并确保 output_dim 与网格/通道一致；
    - 若希望自定义固定 seq_len，不提供 input_hw，统一以 seq_len 控制切分与位置编码。
- config_pdebench.yaml（示例）：
  - model: input_dim: 16384, output_dim: 16384（与 128×128 网格一致）。
  - 建议修正：将 hidden_dim 更名为 d_model；补齐 seq_len 或 input_hw、output_head_type、out_channels_per_token 等必要字段以确保 forward 可计算且输出维度匹配目标。

## 6. 参数规模与头部策略建议
- Pressure（如 20×20）：
  - 建议：input_hw=[20,20] ⇒ seq_len=400；若需要回归到整张网格，推荐 per_token 头并设置 out_channels_per_token=1，使 output_dim=400（或输出多个通道则为 400×C）。若必须输出展平大向量（例如 40000），需在解码/输出映射中确认映射逻辑与目标一致。
- PDEBench（128×128 ⇒ seq_len=16384）：
  - 建议：优先使用 global 头减少参数，或使用稀疏/低秩映射；若使用 per_token 头，务必控制 d_model、num_layers、num_heads，避免显存/参数爆炸。
  - 推荐范围：d_model 256–512、num_layers 4–8、num_heads 4–8（按显存调整），并谨慎设置 batch_size。

## 7. 形状断言与快速校验建议
- 训练循环断言（插入到获取 batch 后，forward 前）：
  - in_press: [B, input_dim]
  - out_pressure: [B, output_dim]
  - time_steps: [B] 或 [B, 1]
  - pred 与 out_pressure 形状一致（global 头为 [B, output_dim]；per_token 头时根据约定 flatten/reshape 后再比对）。
- 建议在 trainer 中加入 try/except + 断言失败时打印实际形状，便于快速定位。
- 最小化 smoke test：构造 dummy 输入（匹配当前配置的 input_dim、time_steps、seq_len/input_hw）直接走一次前向与损失，记录 pred.shape 与 loss 值。

## 8. 已确认的关键要点（Checklist）
- 数据集输出与 DataLoader 批次形状与模型 forward 期望一致。
- 模型 forward 中：embedding → reshape 到 [B, seq_len, d_model]，叠加时间与位置编码，再经编码/解码与输出头得到最终张量。
- output_head_type：
  - global：输出 [B, output_dim]；
  - per_token：输出 [B, seq_len, out_channels_per_token]，与目标对齐时需满足 output_dim = seq_len × out_channels_per_token 或在损失前做一致性变换。
- 配置项优先级：提供 input_hw 时，实际 seq_len = H×W 将覆盖显式 seq_len。

## 9. 下一步可执行事项
1) 在 training/trainer.py 的 train_model 训练循环中加入形状断言与失败时的详细日志。
2) 为 PDEBench 的 config_pdebench.yaml 修正/补齐 model 字段（将 hidden_dim 更名为 d_model，并补齐 output_head_type 等）。
3) 提供一个 tests/test_shapes_smoke.py（或脚本）进行 dummy 形状快速验证，纳入 CI 或本地预检。

如需我直接落地上述断言与 smoke test，请告知：
- 使用数据集（pressure / pdebench / toy）
- 目标头部类型（global / per_token）
- 希望的最终输出形状（例如 [B, 16384] 或 [B, 16384, 1]）

## 10. 数据集可改进点（按优先级）

高优先级（影响稳定性与泛化）
- 标准化与单位一致：为各变量计算/保存全库 mean/std，训练与推理使用同一组；统一物理单位，避免跨场景尺度不一致。
- 时间维度统一与缩放：将 time_step 归一化到 [0,1] 或标准化到 N(0,1)，并保存时间范围元数据，确保与模型的时间编码严格对齐。
- 划分策略与可复现：基于场景/时间段分层划分 train/val/test；固定随机种子并固化划分清单，防止数据泄漏。
- 质量控制与异常清洗：扫描 NaN/Inf、极端异常、全零帧；生成数据质量报告（数值范围、缺失率、边界条件统计）。

中优先级（提升信息与效率）
- 网格与坐标信息保留：Dataset 返回 (x,y[,z]) 坐标或归一化索引；对大分辨率样本支持 patchify（如 8×8/16×16）并返回 patch grid。
- 多变量与掩码：多模态对齐为多通道输出；对无效/边界区域提供 mask，用于损失加权与推理跳过。
- 数据加载与性能：采用 HDF5 chunk/NPY memmap；DataLoader 启用 pin_memory、persistent_workers、合适的 prefetch_factor；配置 worker_init_fn 固定随机种子；对常用数据做本地缓存。
- 统计与报告：data_report 脚本输出分布直方图、时间覆盖率、边界条件比例等，为超参与损失设计提供依据。

加分项（提升鲁棒性与适配）
- 课程式采样与时间覆盖：从“容易时间段”逐步过渡到“困难时间段”，保证不同时间尺度的覆盖。
- 物理友好增强：小噪声、边界条件微扰、局部随机 mask（Cutout-style），注意不破坏物理约束。
- 版本与元数据：生成 schema 版本与元数据 JSON（坐标系、单位、统计、划分、checksum）。

## 11. 模型是否必须严格遵循 AIAYN？

不需要。可依据任务特性进行有针对性的改造，只要训练稳定、可泛化并契合需求即可：
- 大分辨率网格（如 128×128，seq_len=16384）：优先 Query-based/Perceiver 式解码（少量查询 + 交叉注意力），或 Patchify + 分层/轴向注意力，结合低秩/近似注意力以控显存。
- 中小分辨率（如 20×20，seq_len=400）：per_token 头可行；输出整网格时令 out_channels_per_token=通道数使 output_dim=H×W×C；可混合 Conv stem + Transformer。
- 面向物理的增强：Neural Operator（FNO/UNO）与注意力融合、状态空间模型（S4/Mamba）处理超长序列、加入守恒/能量/边界等物理正则。

## 12. 路线图（可落地事项）
- 数据侧：在预处理阶段保存 mean/std、单位、坐标/时间范围；Dataset 返回 (coords, t_norm, mask)；为高分辨率数据加入可选 patchify 管道。
- 模型侧：新增“query-based 输出头”（交叉注意力解码器），复用现有 encoder；对 per_token 模式确保 output_dim 与 seq_len×通道自然对齐，必要时采用分块/低秩投影。
- 工具侧：新增 data_report.py 生成统计/质量摘要；优化 DataLoader（pin_memory、persistent_workers、prefetch_factor）。
- 执行顺序建议：数据标准化与时间归一化 → 形状断言与 smoke test →（对大分辨率）引入 query-based 头与 patchify → 数据报告/监控闭环。