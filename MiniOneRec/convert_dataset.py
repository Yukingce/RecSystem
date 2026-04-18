#!/usr/bin/env python3
"""
Convert dataset (Office/Industrial_and_Scientific) to MiniOneRec format with semantic IDs
"""
"""
convert_dataset.py 的作用是：
    把原始 Amazon 数据（含 item 元信息、交互序列、semantic index）转换成 
    MiniOneRec 可直接训练/评估的目录与文件格式。

主要做了这几件事：
    读取输入文件（同一 data_dir 下）
        *.item.json：item 元信息（title 等）
        *.index.json：item_id -> semantic tokens 映射
        *.train.inter / *.valid.inter / *.test.inter：交互数据
    把 semantic tokens（例如 ["<a_1>", "<b_2>", ...]）拼成单个 semantic_id 字符串（不加空格，直接拼接）。
    生成 info 文件：sid \t title \t item_id，用于 sid 与原始物品的对照。
    把每个 split 转成 CSV，包含 MiniOneRec 需要的字段：
        user_id, history_item_title, item_title, history_item_id, item_id, history_item_sid, item_sid
    可选逻辑：
        --keep_longest_only：训练集每个用户只保留最长历史序列
        --max_valid_samples / --max_test_samples：对 valid/test 进行随机下采样（用 --seed 固定随机性）
    最终输出目录结构是 output_dir/{train,valid,test,info}/...，文件名形如 "{category}_5_2016-10-2018-11.csv" 与 .txt。

    一句话总结：这是把“原始交互 + RQ 生成的语义 ID”打包成 MiniOneRec 下游训练格式的转换脚本。
"""


import json
import pandas as pd
import numpy as np
import os
from typing import Dict, List, Any
import argparse

def load_dataset(data_dir: str, dataset_name: str) -> Dict[str, Any]:
    """Load all dataset files"""
    data = {}
    
    # Load item metadata (id -> {title, description, ...})
    with open(os.path.join(data_dir, f'{dataset_name}.item.json'), 'r') as f:
        data['items'] = json.load(f)
    
    # Load item_id to semantic tokens mapping from index.json
    with open(os.path.join(data_dir, f'{dataset_name}.index.json'), 'r') as f:
        data['item_to_semantic'] = json.load(f)
    
    # Load train/valid/test splits
    splits = {}
    for split in ['train', 'valid', 'test']:
        split_file = os.path.join(data_dir, f'{dataset_name}.{split}.inter')
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                lines = f.readlines()[1:]  # Skip header
                splits[split] = [line.strip().split('\t') for line in lines if line.strip()]
    
    data['splits'] = splits
    return data

def semantic_tokens_to_id(tokens: List[str]) -> str:
    """Convert semantic tokens list to concatenated string with brackets preserved"""
    # Keep brackets and concatenate directly (no spaces)
    return ''.join(tokens)

def create_item_info_file(items: Dict[str, Dict], item_to_semantic: Dict[str, List], output_path: str):
    """Create item info file (sid -> title -> item_id mapping)"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for item_id, item_data in items.items():
            # Get semantic ID from index mapping
            if item_id in item_to_semantic:
                semantic_tokens = item_to_semantic[item_id]
                semantic_id = semantic_tokens_to_id(semantic_tokens)
                # Get item title, fallback to Item_id if not available
                item_title = item_data.get('title', f'Item_{item_id}')
                f.write(f"{semantic_id}\t{item_title}\t{item_id}\n")

def convert_interactions_to_csv(splits: Dict[str, List], items: Dict[str, Dict], 
                               item_to_semantic: Dict[str, List], output_dir: str, category: str = "Office_Products",
                               max_valid_samples: int = None, max_test_samples: int = None, seed: int = 42,
                               keep_longest_only: bool = True):
    """Convert interaction data to MiniOneRec CSV format using semantic IDs"""
    
    import random
    random.seed(seed)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for split_name, split_data in splits.items():
        rows = []
        user_to_longest = {}  # For train data: keep only longest sequence per user
        
        for line in split_data:
            if len(line) != 3:
                continue
                
            user_id, item_sequence, target_item = line
            
            # Parse item sequence - these are item_ids from the interaction data
            if item_sequence.strip():
                history_item_ids = [int(x) for x in item_sequence.split()]
            else:
                history_item_ids = []
            
            target_item_id = int(target_item)
            
            # Convert item_ids to semantic_ids using index mapping
            history_semantic_ids = []
            for item_id in history_item_ids:
                if str(item_id) in item_to_semantic:
                    semantic_tokens = item_to_semantic[str(item_id)]
                    semantic_id = semantic_tokens_to_id(semantic_tokens)
                    history_semantic_ids.append(semantic_id)
            
            # Get target semantic_id
            target_semantic_id = None
            if str(target_item_id) in item_to_semantic:
                semantic_tokens = item_to_semantic[str(target_item_id)]
                target_semantic_id = semantic_tokens_to_id(semantic_tokens)
            
            if target_semantic_id is None:
                continue  # Skip if no semantic_id found for target
            
            # Get item titles using item_ids (not semantic_ids)
            history_item_titles = []
            for item_id in history_item_ids:
                if str(item_id) in items:
                    title = items[str(item_id)].get('title', f'Item_{item_id}')
                    history_item_titles.append(title)
            
            # Target item title
            target_title = items.get(str(target_item_id), {}).get('title', f'Item_{target_item_id}')
            
            # Create row with required fields
            row = {
                'user_id': f'A{user_id}',  # Format like A2013JDMPUV6D9
                'history_item_title': history_item_titles,
                'item_title': target_title,
                'history_item_id': history_item_ids,  # Original item_ids
                'item_id': target_item_id,  # Original item_id
                'history_item_sid': history_semantic_ids,  # Semantic IDs (with brackets)
                'item_sid': target_semantic_id  # Target semantic ID (with brackets)
            }
            
            # For train data: keep only longest sequence per user (if enabled)
            if split_name == 'train' and keep_longest_only:
                sequence_length = len(history_item_ids)
                if user_id not in user_to_longest or sequence_length > len(user_to_longest[user_id]['history_item_id']):
                    user_to_longest[user_id] = row
            else:
                rows.append(row)
        
        # For train data, add only the longest sequences (if enabled)
        if split_name == 'train' and keep_longest_only:
            rows = list(user_to_longest.values())
        
        # Apply sample limits for valid/test sets
        original_count = len(rows)
        if split_name == 'valid' and max_valid_samples is not None and len(rows) > max_valid_samples:
            rows = random.sample(rows, max_valid_samples)
            print(f"  Sampled {max_valid_samples} from {original_count} validation samples")
        elif split_name == 'test' and max_test_samples is not None and len(rows) > max_test_samples:
            rows = random.sample(rows, max_test_samples)
            print(f"  Sampled {max_test_samples} from {original_count} test samples")
        
        # Save to CSV
        if rows:
            df = pd.DataFrame(rows)
            output_file = os.path.join(output_dir, f'{category}_5_2016-10-2018-11.csv')
            df.to_csv(output_file, index=False)
            print(f"Created {split_name} file: {output_file} with {len(rows)} rows")
            if split_name == 'train' and keep_longest_only:
                print(f"  Kept longest sequences for {len(rows)} unique users")
            elif split_name == 'train':
                print(f"  Kept all sequences for train data")
            if rows:
                print(f"  Sample row:")
                print(f"    user_id: {rows[0]['user_id']}")
                print(f"    history_item_id: {rows[0]['history_item_id']}")
                print(f"    item_id: {rows[0]['item_id']}")
                print(f"    history_item_sid: {rows[0]['history_item_sid']}")
                print(f"    item_sid: {rows[0]['item_sid']}")
                print(f"    item_title: {rows[0]['item_title'][:50]}...")

def main():
    parser = argparse.ArgumentParser(description='Convert dataset (Office_Products/Industrial_and_Scientific) to MiniOneRec format with semantic IDs')
    parser.add_argument('--data_dir', type=str, 
                       help='Path to dataset directory')
    parser.add_argument('--dataset_name', type=str, default='Industrial_and_Scientific',
                       help='Dataset name (Office_Products, Industrial_and_Scientific)')
    parser.add_argument('--output_dir', type=str,
                       help='Output directory for MiniOneRec format data')
    parser.add_argument('--category', type=str, default=None,
                       help='Category name for output files (if None, will use dataset_name)')
    parser.add_argument('--max_valid_samples', type=int, default=None,
                       help='Maximum number of samples to keep in validation set (None for all)')
    parser.add_argument('--max_test_samples', type=int, default=None,
                       help='Maximum number of samples to keep in test set (None for all)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for sampling')
    parser.add_argument('--keep_longest_only', action='store_true', default=False,
                       help='Keep only longest sequence per user in train data (default: False)')
    
    args = parser.parse_args()
    
    # Default category to dataset name if not specified
    if args.category is None:
        args.category = args.dataset_name
    
    print(f"Loading {args.dataset_name} data from {args.data_dir}")
    dataset_data = load_dataset(args.data_dir, args.dataset_name)
    
    print(f"Found {len(dataset_data['items'])} items")
    print(f"Found {len(dataset_data['item_to_semantic'])} item-to-semantic mappings")
    
    # Sample a few semantic ID conversions
    sample_items = list(dataset_data['item_to_semantic'].items())[:3]
    for item_id, tokens in sample_items:
        semantic_id = semantic_tokens_to_id(tokens)
        print(f"  Item {item_id}: {tokens} -> {semantic_id}")
    
    # Create output directories
    for subdir in ['train', 'valid', 'test', 'info']:
        os.makedirs(os.path.join(args.output_dir, subdir), exist_ok=True)
    
    # Create item info file
    info_file = os.path.join(args.output_dir, 'info', f'{args.category}_5_2016-10-2018-11.txt')
    create_item_info_file(dataset_data['items'], dataset_data['item_to_semantic'], info_file)
    print(f"Created item info file: {info_file}")
    
    # Convert each split
    for split_name in ['train', 'valid', 'test']:
        if split_name in dataset_data['splits']:
            split_output_dir = os.path.join(args.output_dir, split_name)
            convert_interactions_to_csv(
                {split_name: dataset_data['splits'][split_name]}, 
                dataset_data['items'],
                dataset_data['item_to_semantic'],
                split_output_dir,
                args.category,
                max_valid_samples=args.max_valid_samples,
                max_test_samples=args.max_test_samples,
                seed=args.seed,
                keep_longest_only=args.keep_longest_only
            )
    
    print(f"\nConversion completed! Data saved to {args.output_dir}")
    print(f"You can now use these files with MiniOneRec training scripts by setting category='{args.category}'")
    
    # Show sampling information if limits were applied
    if args.max_valid_samples is not None:
        print(f"Validation set was limited to {args.max_valid_samples} samples")
    if args.max_test_samples is not None:
        print(f"Test set was limited to {args.max_test_samples} samples")

if __name__ == '__main__':
    main()
