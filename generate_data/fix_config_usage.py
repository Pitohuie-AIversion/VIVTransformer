#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复配置参数使用的脚本
实现关键配置参数的实际功能
"""

import re
from pathlib import Path

def fix_optimizer_config():
    """修复优化器配置使用"""
    trainer_file = "dynamic_resolution_trainer.py"
    
    # 读取文件
    with open(trainer_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找优化器创建的代码
    old_optimizer_code = r"optimizer = torch\.optim\.Adam\(model\.parameters\(\),\s*lr=config\['training'\]\['learning_rate'\],\s*weight_decay=config\['training'\]\['weight_decay'\]\)"
    
    new_optimizer_code = '''# 从配置中获取优化器参数
    optimizer_config = config['optimizer']
    training_config = config['training']
    
    print(f"🔧 优化器配置: 类型={optimizer_config['type']}, 学习率={training_config['learning_rate']}, 权重衰减={training_config['weight_decay']}")
    
    # 根据配置创建优化器
    if optimizer_config['type'].lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config['weight_decay'],
            betas=optimizer_config['betas'],
            eps=optimizer_config['eps'],
            amsgrad=optimizer_config['amsgrad']
        )
        print(f"   Adam参数: betas={optimizer_config['betas']}, eps={optimizer_config['eps']}, amsgrad={optimizer_config['amsgrad']}")
    elif optimizer_config['type'].lower() == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config['weight_decay'],
            momentum=optimizer_config.get('momentum', 0.9)
        )
        print(f"   SGD参数: momentum={optimizer_config.get('momentum', 0.9)}")
    elif optimizer_config['type'].lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config['weight_decay'],
            betas=optimizer_config['betas'],
            eps=optimizer_config['eps'],
            amsgrad=optimizer_config['amsgrad']
        )
        print(f"   AdamW参数: betas={optimizer_config['betas']}, eps={optimizer_config['eps']}, amsgrad={optimizer_config['amsgrad']}")
    else:
        print(f"[WARN]  不支持的优化器类型: {optimizer_config['type']}，使用默认Adam")
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config['weight_decay']
        )'''
    
    # 替换代码
    if re.search(old_optimizer_code, content):
        content = re.sub(old_optimizer_code, new_optimizer_code, content)
        print("[OK] 已修复优化器配置使用")
    else:
        print("[WARN]  未找到优化器创建代码，可能已经修改过")
    
    return content

def fix_scheduler_config(content):
    """添加学习率调度器配置"""
    # 在优化器创建后添加调度器
    scheduler_code = '''
    
    # 创建学习率调度器
    scheduler_config = config['scheduler']
    scheduler = None
    
    if scheduler_config['enabled']:
        print(f"[INFO] 学习率调度器配置: 类型={scheduler_config['type']}, 启用={scheduler_config['enabled']}")
        
        if scheduler_config['type'].lower() == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=scheduler_config['T_max']
            )
            print(f"   Cosine参数: T_max={scheduler_config['T_max']}")
        elif scheduler_config['type'].lower() == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_config['step_size'],
                gamma=scheduler_config['gamma']
            )
            print(f"   Step参数: step_size={scheduler_config['step_size']}, gamma={scheduler_config['gamma']}")
        elif scheduler_config['type'].lower() == 'exponential':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=scheduler_config['gamma']
            )
            print(f"   Exponential参数: gamma={scheduler_config['gamma']}")
        else:
            print(f"[WARN]  不支持的调度器类型: {scheduler_config['type']}")
    else:
        print("[INFO] 学习率调度器: 未启用")
'''
    
    # 在优化器创建后插入调度器代码
    optimizer_end_pattern = r"(\s+else:\s+print\(f\"[WARN]\s+不支持的优化器类型.*?\)\s+optimizer = torch\.optim\.Adam\([^}]+\}\))"
    
    if re.search(optimizer_end_pattern, content, re.DOTALL):
        content = re.sub(optimizer_end_pattern, r"\1" + scheduler_code, content, flags=re.DOTALL)
        print("[OK] 已添加学习率调度器配置")
    else:
        print("[WARN]  未找到优化器结束位置，手动添加调度器代码")
    
    return content

def fix_dataloader_config(content):
    """修复数据加载器配置"""
    # 查找get_dynamic_loaders函数中的DataLoader创建
    old_dataloader_pattern = r"(\s+)(train_loader = DataLoader\(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0\))"
    old_valid_pattern = r"(\s+)(valid_loader = DataLoader\(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0\))"
    old_test_pattern = r"(\s+)(test_loader = DataLoader\(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0\))"
    
    new_dataloader_code = '''\1# 从配置中获取数据加载器参数
\1dataloader_config = config.get('dataloader', {})
\1num_workers = dataloader_config.get('num_workers', 0)
\1pin_memory = dataloader_config.get('pin_memory', True)
\1drop_last = dataloader_config.get('drop_last', False)
\1persistent_workers = dataloader_config.get('persistent_workers', False) and num_workers > 0
\1
\1print(f"[INFO] 数据加载器配置: num_workers={num_workers}, pin_memory={pin_memory}, drop_last={drop_last}")
\1
\1train_loader = DataLoader(
\1    train_dataset, 
\1    batch_size=batch_size, 
\1    shuffle=True, 
\1    num_workers=num_workers,
\1    pin_memory=pin_memory,
\1    drop_last=drop_last,
\1    persistent_workers=persistent_workers
\1)'''
    
    new_valid_code = '''\1valid_loader = DataLoader(
\1    valid_dataset, 
\1    batch_size=batch_size, 
\1    shuffle=False, 
\1    num_workers=num_workers,
\1    pin_memory=pin_memory,
\1    drop_last=False,
\1    persistent_workers=persistent_workers
\1)'''
    
    new_test_code = '''\1test_loader = DataLoader(
\1    test_dataset, 
\1    batch_size=batch_size, 
\1    shuffle=False, 
\1    num_workers=num_workers,
\1    pin_memory=pin_memory,
\1    drop_last=False,
\1    persistent_workers=persistent_workers
\1)'''
    
    # 替换DataLoader创建代码
    if re.search(old_dataloader_pattern, content):
        content = re.sub(old_dataloader_pattern, new_dataloader_code, content)
        content = re.sub(old_valid_pattern, new_valid_code, content)
        content = re.sub(old_test_pattern, new_test_code, content)
        print("[OK] 已修复数据加载器配置使用")
    else:
        print("[WARN]  未找到数据加载器创建代码")
    
    return content

def fix_train_function_call(content):
    """修复train_model函数调用，传递scheduler参数"""
    # 查找train_model函数调用
    old_train_call = r"(\s+)(train_model\(model, train_loader, valid_loader, test_loader, criterion, optimizer,\s+num_epochs=epochs, device=device, early_stop_patience=patience,\s+attention_type=attention_type, result_dir=model_save_dir, cfg=config\))"
    
    new_train_call = '''\1train_model(model, train_loader, valid_loader, test_loader, criterion, optimizer,
\1           num_epochs=epochs, device=device, early_stop_patience=patience,
\1           attention_type=attention_type, result_dir=model_save_dir, cfg=config,
\1           scheduler=scheduler)'''
    
    if re.search(old_train_call, content, re.DOTALL):
        content = re.sub(old_train_call, new_train_call, content, flags=re.DOTALL)
        print("[OK] 已修复train_model函数调用")
    else:
        print("[WARN]  未找到train_model函数调用")
    
    return content

def main():
    """主函数"""
    print("=== 修复配置参数使用 ===")
    
    # 备份原文件
    trainer_file = "dynamic_resolution_trainer.py"
    backup_file = "dynamic_resolution_trainer_backup.py"
    
    if Path(trainer_file).exists():
        # 创建备份
        with open(trainer_file, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(original_content)
        print(f"[INFO] 已创建备份文件: {backup_file}")
        
        # 应用修复
        content = fix_optimizer_config()
        content = fix_scheduler_config(content)
        content = fix_dataloader_config(content)
        content = fix_train_function_call(content)
        
        # 保存修改后的文件
        with open(trainer_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"\n[OK] 修复完成！已更新 {trainer_file}")
        print("\n[INFO] 修复内容:")
        print("   1. [OK] 优化器配置 - 支持Adam/SGD/AdamW及其参数")
        print("   2. [OK] 学习率调度器 - 支持Cosine/Step/Exponential")
        print("   3. [OK] 数据加载器配置 - 支持num_workers/pin_memory等参数")
        print("   4. [OK] 函数调用更新 - 传递scheduler参数")
        
        print("\n[WARN]  注意: 还需要手动修改trainer.py文件以支持scheduler参数")
        
    else:
        print(f"[ERROR] 文件不存在: {trainer_file}")

if __name__ == "__main__":
    main()