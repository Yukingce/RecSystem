import argparse
import collections
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from datasets import EmbDataset  # noqa: E402
from models.rqvae import RQVAE  # noqa: E402


def resolve_existing_path(path_str):
    if os.path.isabs(path_str):
        return path_str

    candidates = [
        os.path.abspath(path_str),
        os.path.abspath(os.path.join(SCRIPT_DIR, path_str)),
        os.path.abspath(os.path.join(os.path.dirname(SCRIPT_DIR), path_str)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate Collision Rate from a trained RQVAE checkpoint."
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/media/rxjqr/wy/MiniOneRec/rq/output/Industrial_and_Scientific_rqvae/Apr-19-2026_14-52-32_3_64/best_collision_model.pth",
        help="Path to the RQVAE checkpoint.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="",
        help="Optional embedding path override. If empty, use checkpoint args.data_path.",
    )
    parser.add_argument("--batch_size", type=int, default=2048, help="Inference batch size.")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of dataloader workers."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device for inference, e.g. cuda:0 or cpu.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="collision_rate_result.json",
        help="Output filename to save in checkpoint folder.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["raw", "post", "both"],
        default="raw",
        help="raw: original collision rate; post: after collision handling; both: compute both.",
    )
    parser.add_argument(
        "--post_max_rounds",
        type=int,
        default=20,
        help="Max rounds for post collision handling (same idea as generate_indices.py).",
    )
    return parser.parse_args()


def get_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        if "state_dict" in ckpt_obj:
            return ckpt_obj["state_dict"]
        if "model_state_dict" in ckpt_obj:
            return ckpt_obj["model_state_dict"]
    return ckpt_obj


def build_prefix(num_levels):
    letters = "abcdefghijklmnopqrstuvwxyz"
    prefix = []
    for i in range(num_levels):
        tag = letters[i] if i < len(letters) else f"l{i}"
        prefix.append(f"<{tag}_{{}}>")
    return prefix


def to_rqvae_code_str(index_array):
    return "-".join([str(int(x)) for x in index_array])


def check_collision(all_indices_str):
    return len(all_indices_str) == len(set(all_indices_str.tolist()))


def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count


def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)
    return [ids for ids in index2id.values() if len(ids) > 1]


def calculate_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str.tolist()))
    collision_rate = (tot_item - tot_indice) / tot_item if tot_item > 0 else 0.0
    max_conflicts = max(get_indices_count(all_indices_str).values()) if tot_item > 0 else 0
    return {
        "total_items": int(tot_item),
        "unique_indices": int(tot_indice),
        "collision_rate": float(collision_rate),
        "max_conflicts": int(max_conflicts),
    }


def collect_all_indices(model, data_loader, device):
    all_indices_np = []
    all_indices_str = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Computing indices (use_sk=False)"):
            batch = batch.to(device)
            indices = model.get_indices(batch, use_sk=False)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for index in indices:
                all_indices_np.append(index.astype(np.int64))
                # Strictly follow rq/trainer.py: "-".join(str(int(_)) for _ in index)
                all_indices_str.append(to_rqvae_code_str(index))
    return np.array(all_indices_np), np.array(all_indices_str)


def apply_post_collision_handling(model, dataset, device, all_indices_np, all_indices_str, max_rounds):
    # Keep the same handling strategy as rq/generate_indices.py
    prefix = build_prefix(all_indices_np.shape[1])
    for vq in model.rq.vq_layers[:-1]:
        vq.sk_epsilon = 0.0
    if model.rq.vq_layers[-1].sk_epsilon == 0.0:
        model.rq.vq_layers[-1].sk_epsilon = 0.003

    rounds = 0
    while rounds < max_rounds and not check_collision(all_indices_str):
        collision_item_groups = get_collision_item(all_indices_str)
        for collision_items in collision_item_groups:
            d = dataset[collision_items].to(device)
            indices = model.get_indices(d, use_sk=True)
            indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
            for item, index in zip(collision_items, indices):
                all_indices_np[item] = index.astype(np.int64)
                # Update to same style as raw/rqvae.py for post collision re-statistics
                all_indices_str[item] = to_rqvae_code_str(index)
        rounds += 1

    # Also return generate_indices.py style code strings for reference consistency check.
    all_indices_sid_str = np.array(
        [str([prefix[i].format(int(ind)) for i, ind in enumerate(index)]) for index in all_indices_np]
    )
    return all_indices_np, all_indices_str, all_indices_sid_str, rounds


def main():
    args = parse_args()

    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", None) if isinstance(ckpt, dict) else None
    state_dict = get_state_dict(ckpt)

    if ckpt_args is None:
        raise ValueError("Checkpoint does not contain `args`, cannot rebuild RQVAE config.")

    data_path_raw = args.data_path.strip() if args.data_path.strip() else ckpt_args.data_path
    data_path = resolve_existing_path(data_path_raw)
    dataset = EmbDataset(data_path)

    model = RQVAE(
        in_dim=dataset.dim,
        num_emb_list=ckpt_args.num_emb_list,
        e_dim=ckpt_args.e_dim,
        layers=ckpt_args.layers,
        dropout_prob=ckpt_args.dropout_prob,
        bn=ckpt_args.bn,
        loss_type=ckpt_args.loss_type,
        quant_loss_weight=ckpt_args.quant_loss_weight,
        beta=getattr(ckpt_args, "beta", 0.25),
        kmeans_init=ckpt_args.kmeans_init,
        kmeans_iters=ckpt_args.kmeans_iters,
        sk_epsilons=ckpt_args.sk_epsilons,
        sk_iters=ckpt_args.sk_iters,
    )
    model.load_state_dict(state_dict, strict=True)

    device = torch.device(args.device)
    model = model.to(device)
    model.eval()

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    raw_indices, raw_indices_str = collect_all_indices(model=model, data_loader=data_loader, device=device)
    raw_result = calculate_collision(raw_indices_str)

    output_dir = os.path.dirname(os.path.abspath(args.ckpt_path))
    output_path = os.path.join(output_dir, args.output_name)
    result = {}
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            try:
                result = json.load(f)
            except json.JSONDecodeError:
                result = {}

    result.update({
        "ckpt_path": os.path.abspath(args.ckpt_path),
        "data_path": os.path.abspath(data_path),
        "formula": "(tot_item - tot_indice) / tot_item",
        "compatible_with": "rq/generate_indices.py",
    })

    if args.mode in ("raw", "both"):
        result["total_items"] = raw_result["total_items"]
        result["unique_indices"] = raw_result["unique_indices"]
        result["collision_rate"] = raw_result["collision_rate"]
        result["max_conflicts"] = raw_result["max_conflicts"]
        result["raw_calculated_at"] = datetime.now().isoformat(timespec="seconds")

    if args.mode in ("post", "both"):
        post_indices = raw_indices.copy()
        post_indices_str = raw_indices_str.copy()
        post_indices, post_indices_str, post_indices_sid_str, post_rounds = apply_post_collision_handling(
            model=model,
            dataset=dataset,
            device=device,
            all_indices_np=post_indices,
            all_indices_str=post_indices_str,
            max_rounds=args.post_max_rounds,
        )
        post_result = calculate_collision(post_indices_str)
        post_sid_style_result = calculate_collision(post_indices_sid_str)
        result["post_collision_handling_rounds"] = int(post_rounds)
        result["post_unique_indices"] = post_result["unique_indices"]
        result["post_collision_rate"] = post_result["collision_rate"]
        result["post_max_conflicts"] = post_result["max_conflicts"]
        result["post_sid_style_collision_rate"] = post_sid_style_result["collision_rate"]
        result["post_calculated_at"] = datetime.now().isoformat(timespec="seconds")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Mode: {args.mode}")
    if args.mode in ("raw", "both"):
        print(
            f"Raw Collision Rate: {raw_result['collision_rate']:.10f} "
            f"(items={raw_result['total_items']}, unique={raw_result['unique_indices']})"
        )
    if args.mode in ("post", "both"):
        print(
            f"Post Collision Rate: {result['post_collision_rate']:.10f} "
            f"(unique={result['post_unique_indices']}, rounds={result['post_collision_handling_rounds']})"
        )
    print(f"Saved result to: {output_path}")


if __name__ == "__main__":
    main()
