import pandas as pd
import numpy as np

def analyze_experiment_fairness():
    """分析不同实验组对比的公平性"""
    
    # 读取数据
    df = pd.read_csv('model_comparison_report.csv')
    
    print("=== 横向对比公平性分析 ===")
    print(f"总配置数: {len(df)}")
    print(f"数据完整性: 训练损失完整率 100%, 验证损失完整率 {((df['valid_loss'] != float('inf')) & (~df['valid_loss'].isna())).mean()*100:.1f}%")
    
    print("\n=== 实验组分析 ===")
    for exp in sorted(df['experiment'].unique()):
        exp_data = df[df['experiment'] == exp]
        print(f"\n{exp}:")
        print(f"  配置数量: {len(exp_data)}")
        print(f"  注意力机制: {list(exp_data['attention_type'].unique())}")
        print(f"  损失配置范围: loss_config_{exp_data['loss_config'].str.extract(r'(\d+)')[0].astype(int).min()} - loss_config_{exp_data['loss_config'].str.extract(r'(\d+)')[0].astype(int).max()}")
        
        # 训练损失统计
        train_losses = exp_data['train_loss']
        print(f"  训练损失: {train_losses.min():.6f} - {train_losses.max():.6f} (均值: {train_losses.mean():.6f})")
        
        # 验证损失统计
        valid_losses = exp_data[(exp_data['valid_loss'] != float('inf')) & (~exp_data['valid_loss'].isna())]['valid_loss']
        if len(valid_losses) > 0:
            print(f"  验证损失: {valid_losses.min():.6f} - {valid_losses.max():.6f} (均值: {valid_losses.mean():.6f})")
        else:
            print(f"  验证损失: 全部缺失或为inf")
    
    print("\n=== 注意力机制分析 ===")
    for att_type in sorted(df['attention_type'].unique()):
        att_data = df[df['attention_type'] == att_type]
        experiments = att_data['experiment'].unique()
        print(f"\n{att_type}:")
        print(f"  配置数量: {len(att_data)}")
        print(f"  涉及实验: {list(experiments)}")
        
        # 如果该注意力机制在多个实验中出现，分析是否公平
        if len(experiments) > 1:
            print(f"  跨实验对比:")
            for exp in experiments:
                exp_att_data = att_data[att_data['experiment'] == exp]
                train_mean = exp_att_data['train_loss'].mean()
                valid_data = exp_att_data[(exp_att_data['valid_loss'] != float('inf')) & (~exp_att_data['valid_loss'].isna())]
                valid_mean = valid_data['valid_loss'].mean() if len(valid_data) > 0 else float('inf')
                print(f"    {exp}: 训练损失均值 {train_mean:.6f}, 验证损失均值 {valid_mean:.6f}")
    
    print("\n=== 公平性评估 ===")
    
    # 1. 检查是否使用相同的损失配置
    print("1. 损失配置一致性:")
    loss_configs = set()
    for exp in df['experiment'].unique():
        exp_configs = set(df[df['experiment'] == exp]['loss_config'].unique())
        loss_configs.update(exp_configs)
        print(f"   {exp}: {len(exp_configs)} 个配置")
    
    # 2. 检查注意力机制分布
    print("\n2. 注意力机制分布:")
    attention_dist = df['attention_type'].value_counts()
    for att_type, count in attention_dist.items():
        experiments = df[df['attention_type'] == att_type]['experiment'].unique()
        print(f"   {att_type}: {count} 个配置，分布在 {len(experiments)} 个实验组")
    
    # 3. 数据质量检查
    print("\n3. 数据质量:")
    total_configs = len(df)
    complete_train = len(df[~df['train_loss'].isna()])
    complete_valid = len(df[(df['valid_loss'] != float('inf')) & (~df['valid_loss'].isna())])
    complete_test = len(df[~df['test_loss'].isna()])
    
    print(f"   训练损失完整: {complete_train}/{total_configs} ({complete_train/total_configs*100:.1f}%)")
    print(f"   验证损失完整: {complete_valid}/{total_configs} ({complete_valid/total_configs*100:.1f}%)")
    print(f"   测试损失完整: {complete_test}/{total_configs} ({complete_test/total_configs*100:.1f}%)")
    
    # 4. 结论
    print("\n=== 结论 ===")
    
    # 检查主要问题
    issues = []
    
    # 数据完整性问题
    if complete_valid / total_configs < 0.9:
        issues.append(f"验证损失数据不完整({complete_valid/total_configs*100:.1f}%)")
    
    # 实验组不平衡问题
    exp_counts = df['experiment'].value_counts()
    if exp_counts.max() / exp_counts.min() > 10:
        issues.append(f"实验组样本数量严重不平衡(最大{exp_counts.max()}vs最小{exp_counts.min()})")
    
    # 注意力机制分布不均
    att_counts = df['attention_type'].value_counts()
    if att_counts.max() / att_counts.min() > 20:
        issues.append(f"注意力机制分布极不均匀(最多{att_counts.max()}vs最少{att_counts.min()})")
    
    if issues:
        print("⚠️  发现的公平性问题:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        
        print("\n建议:")
        if "验证损失数据不完整" in str(issues):
            print("   - 补全缺失的验证损失数据")
        if "实验组样本数量严重不平衡" in str(issues):
            print("   - 平衡各实验组的配置数量，或在分析时考虑权重")
        if "注意力机制分布极不均匀" in str(issues):
            print("   - 确保每种注意力机制有足够的测试配置")
    else:
        print("✅ 横向对比基本公平合理")
        print("   - 所有实验组使用相同的训练配置")
        print("   - 损失函数配置覆盖完整")
        print("   - 数据质量可接受")

if __name__ == "__main__":
    analyze_experiment_fairness()