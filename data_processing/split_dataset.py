#!/usr/bin/env python3
"""
随机取样并划分训练集和测试集
"""
import json
import random
import os
from pathlib import Path


def load_json(file_path):
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, file_path):
    """保存JSON文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def split_dataset(input_file, train_ratio=0.8, random_seed=42):
    """
    随机取样并划分训练集和测试集
    
    Args:
        input_file: 输入的JSON文件路径
        train_ratio: 训练集比例，默认0.8（80%训练集，20%测试集）
        random_seed: 随机种子，确保结果可复现
    """
    # 设置随机种子
    random.seed(random_seed)
    
    # 加载数据
    print(f"正在加载数据: {input_file}")
    data = load_json(input_file)
    
    # 获取所有case的key
    case_keys = list(data.keys())
    total_cases = len(case_keys)
    print(f"总病例数: {total_cases}")
    
    # 随机打乱
    random.shuffle(case_keys)
    
    # 计算划分点
    train_size = int(total_cases * train_ratio)
    train_keys = case_keys[:train_size]
    test_keys = case_keys[train_size:]
    
    print(f"训练集大小: {len(train_keys)} ({len(train_keys)/total_cases*100:.2f}%)")
    print(f"测试集大小: {len(test_keys)} ({len(test_keys)/total_cases*100:.2f}%)")
    
    # 构建训练集和测试集
    train_data = {key: data[key] for key in train_keys}
    test_data = {key: data[key] for key in test_keys}
    
    # 保存结果
    input_path = Path(input_file)
    output_dir = input_path.parent
    
    train_file = output_dir / "train_cases.json"
    test_file = output_dir / "test_cases.json"
    
    print(f"正在保存训练集: {train_file}")
    save_json(train_data, str(train_file))
    
    print(f"正在保存测试集: {test_file}")
    save_json(test_data, str(test_file))
    
    print("数据集划分完成！")
    
    return train_file, test_file


if __name__ == "__main__":
    input_file = "/data/aj/RAG/breast_cancer_planner_agent/data/raw/full_cases.json"
    train_file, test_file = split_dataset(input_file, train_ratio=0.8, random_seed=42)
    print(f"\n训练集: {train_file}")
    print(f"测试集: {test_file}")

