# from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
# import transformers
# import torch
import os
import fire
import math
import json
import pandas as pd
import numpy as np
    
from tqdm import tqdm
def gao(path, item_path):
    if type(path) != list:
        path = [path]
    if item_path.endswith(".txt"):
        item_path = item_path[:-4]
    CC = 0
        
    
    f = open(f"{item_path}.txt", 'r')
    items = f.readlines()
    # item_names = [ _[:-len(_.split('\t')[-1])].strip() for _ in items]
    item_names= [_.split('\t')[0].strip() for _ in items]
    item_ids = [_ for _ in range(len(item_names))]
    item_dict = dict()
    for i in range(len(item_names)):
        if item_names[i] not in item_dict:
            item_dict[item_names[i]] = [item_ids[i]]
        else:   
            item_dict[item_names[i]].append(item_ids[i])
    
    

    result_dict = dict()
    topk_list = [1, 3, 5, 10, 20, 50]
    n_beam = -1
    for p in path:
        result_dict[p] = {
            "NDCG": [],
            "HR": [],
            "AUC_pos_vs_neg": 0.0,
        }
        f = open(p, 'r')
        import json
        test_data = json.load(f)
        f.close()
        
        text = [ [_.strip("\"\n").strip() for _ in sample["predict"]] for sample in test_data]
        
        ALLAUC = 0.0
        for sample_idx, sample in tqdm(enumerate(text)):
            if n_beam == -1:
                n_beam = len(sample)
                valid_topk = [k for k in topk_list if k <= n_beam]
                ALLNDCG = np.zeros(len(valid_topk))
                ALLHR = np.zeros(len(valid_topk))
            if type(test_data[sample_idx]['output']) == list:
                target_item = test_data[sample_idx]['output'][0].strip("\"").strip(" ")
            else:
                target_item = test_data[sample_idx]['output'].strip(" \n\"")
            minID = 1000000
            for i in range(len(sample)):
                
                if sample[i] not in item_dict:
                    CC += 1
                    print(sample[i])
                    print(target_item)
                if sample[i] == target_item:
                    minID = i
                    break

            # Align with test_reward_consistency.py:
            # AUC_pos_vs_neg = mean over negatives of
            #   1.0 if pos_score > neg_score
            #   0.5 if pos_score == neg_score
            #   0.0 otherwise
            # Here we only have ranked candidates (no raw scores), so we use rank-based
            # pairwise comparisons, equivalent to the above when there are no score ties.
            if n_beam > 1 and minID < n_beam:
                pos_rank_idx = minID
                neg_pair_score = 0.0
                for neg_rank_idx in range(n_beam):
                    if neg_rank_idx == pos_rank_idx:
                        continue
                    if pos_rank_idx < neg_rank_idx:
                        neg_pair_score += 1.0
                    elif pos_rank_idx == neg_rank_idx:
                        neg_pair_score += 0.5
                ALLAUC += neg_pair_score / (n_beam - 1)

            for topk_idx, topk in enumerate(topk_list):
                if topk > n_beam:
                    continue
                if minID < topk:
                    ALLNDCG[topk_idx] = ALLNDCG[topk_idx] + (1 / math.log(minID + 2))
                    ALLHR[topk_idx] = ALLHR[topk_idx] + 1
        print(n_beam)
        valid_topk = [k for k in topk_list if k <= n_beam]
        ndcg_values = ALLNDCG / len(text) / (1.0 / math.log(2))
        hr_values = ALLHR / len(text)
        auc_value = ALLAUC / len(text)
        print(valid_topk)
        print(f"NDCG:\t{ndcg_values}")
        print(f"HR\t{hr_values}")
        print(f"AUC_pos_vs_neg\t{auc_value}")
        print(CC)
        result_dict[p]["NDCG"] = ndcg_values.tolist()
        result_dict[p]["HR"] = hr_values.tolist()
        result_dict[p]["AUC_pos_vs_neg"] = float(auc_value)

if __name__=='__main__':
    fire.Fire(gao)
