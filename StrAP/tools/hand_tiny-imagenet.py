import os
import json
import argparse
import numpy as np
from typing import Dict, List, Tuple


DIRICHLET_ALPHA = 0.3


def find_tiny_root(root: str) -> str:
    candidate = os.path.join(root, "tiny-imagenet-200")
    if os.path.isdir(candidate):
        return candidate
    if os.path.isdir(root) and os.path.basename(root.rstrip("/")) == "tiny-imagenet-200":
        return root
    raise FileNotFoundError(
        f"Cannot find 'tiny-imagenet-200' under root={root!r}. "
        f"Please pass --root that contains tiny-imagenet-200/."
    )


def load_train_samples(train_dir: str, label_key: str) -> Tuple[List[dict], Dict[str, int]]:
    wnids = sorted(
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    )
    if len(wnids) == 0:
        raise RuntimeError(f"No class subdirs found in {train_dir}")

    wnid_to_label = {wnid: idx for idx, wnid in enumerate(wnids)}

    samples: List[dict] = []
    exts = (".jpeg", ".jpg", ".png")

    for wnid in wnids:
        img_dir = os.path.join(train_dir, wnid, "images")
        if not os.path.isdir(img_dir):
            img_dir = os.path.join(train_dir, wnid)
        if not os.path.isdir(img_dir):
            continue

        for fn in os.listdir(img_dir):
            if not fn.lower().endswith(exts):
                continue
            img_path = os.path.join(img_dir, fn)
            samples.append({
                "img_path": img_path,
                label_key: wnid_to_label[wnid],
                "wnid": wnid,
            })

    if len(samples) == 0:
        raise RuntimeError(f"No train images found in {train_dir}")

    return samples, wnid_to_label


def load_val_samples(val_dir: str, wnid_to_label: Dict[str, int]) -> List[dict]:
    ann_path = os.path.join(val_dir, "val_annotations.txt")
    if not os.path.isfile(ann_path):
        raise FileNotFoundError(f"val_annotations.txt not found in {val_dir}")

    img_to_wnid: Dict[str, str] = {}
    with open(ann_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            fname, wnid = parts[0], parts[1]
            img_to_wnid[fname] = wnid

    images_dir = os.path.join(val_dir, "images")
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"val/images dir not found in {val_dir}")

    exts = (".jpeg", ".jpg", ".png")
    samples: List[dict] = []

    for fn in os.listdir(images_dir):
        if not fn.lower().endswith(exts):
            continue
        if fn not in img_to_wnid:
            raise KeyError(f"Val image {fn!r} not found in val_annotations.txt.")

        wnid = img_to_wnid[fn]
        if wnid not in wnid_to_label:
            raise KeyError(f"Wnid {wnid!r} in val not found in train classes.")

        label_id = wnid_to_label[wnid]
        img_path = os.path.join(images_dir, fn)
        samples.append({
            "img_path": img_path,
            "label": label_id,  
        })

    if len(samples) == 0:
        raise RuntimeError(f"No val images found in {images_dir}")

    return samples


def dirichlet_split_non_iid(
    samples: List[dict],
    label_key: str,
    num_clients: int,
    alpha: float,
    seed: int,
) -> Dict[str, List[dict]]:
    np.random.seed(seed)

    label_to_indices: Dict[int, List[int]] = {}
    for idx, s in enumerate(samples):
        label_id = int(s[label_key])
        if label_id not in label_to_indices:
            label_to_indices[label_id] = []
        label_to_indices[label_id].append(idx)

    client_buckets: Dict[int, List[int]] = {i: [] for i in range(num_clients)}
    for label_id, idxs in label_to_indices.items():
        idxs = np.array(idxs)
        n = len(idxs)
        if n == 0:
            continue

        proportions = np.random.dirichlet(alpha * np.ones(num_clients))
        counts = np.random.multinomial(n, proportions)

        np.random.shuffle(idxs)

        offset = 0
        for client_id in range(num_clients):
            c = counts[client_id]
            if c <= 0:
                continue
            take = idxs[offset: offset + c]
            offset += c
            client_buckets[client_id].extend(take.tolist())
    federated_dict: Dict[str, List[dict]] = {}
    for client_id in range(num_clients):
        key = f"client_{client_id + 1}"
        federated_dict[key] = []
        idx_list = client_buckets[client_id]
        for idx in idx_list:
            s = samples[idx]
            federated_dict[key].append({
                "img_path": s["img_path"],
                label_key: int(s[label_key]),
            })

    return federated_dict


def save_json(obj, path: str, pretty: bool = False):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if pretty:
            json.dump(obj, f, indent=2, ensure_ascii=False)
        else:
            json.dump(obj, f, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser(description="Convert Tiny-ImageNet to federated classification JSON.")
    ap.add_argument("--root", required=True, help="Root directory containing tiny-imagenet-200/")
    ap.add_argument("--out", default="./datasets/data_cache/tiny_imagenet_train.json",
                    help="Federated training JSON output path")
    ap.add_argument("--val_out", default="./datasets/data_cache/tiny_imagenet_val.json",
                    help="Validation JSON output path")
    ap.add_argument("--num_clients", type=int, default=100, help="Number of federated clients")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pretty", action="store_true", help="Save JSON with indented formatting")
    ap.add_argument("--label-key", default="label",
                    help="Label field name in training JSON (pass --label-key lable if compatibility with 'lable' is needed)")
    ap.add_argument("--save_class_map", default=None,
                    help="Optional: Path to save wnid->id mapping JSON")

    args = ap.parse_args()

    np.random.seed(args.seed)

    tiny_root = find_tiny_root(args.root)
    train_dir = os.path.join(tiny_root, "train")
    val_dir = os.path.join(tiny_root, "val")

    print(f"[INFO] tiny-imagenet-200 root: {tiny_root}")
    print(f"[INFO] train dir: {train_dir}")
    print(f"[INFO] val dir  : {val_dir}")

    train_samples, wnid_to_label = load_train_samples(train_dir, label_key=args.label_key)
    print(f"[INFO] #train samples: {len(train_samples)}, #classes: {len(wnid_to_label)}")

    val_samples = load_val_samples(val_dir, wnid_to_label)
    print(f"[INFO] #val samples: {len(val_samples)}")
    print(f"[INFO] Dirichlet alpha = {DIRICHLET_ALPHA}, num_clients = {args.num_clients}")
    federated_train = dirichlet_split_non_iid(
        samples=train_samples,
        label_key=args.label_key,
        num_clients=args.num_clients,
        alpha=DIRICHLET_ALPHA,
        seed=args.seed,
    )
    total_train = sum(len(v) for v in federated_train.values())
    print(f"[INFO] Total federated train samples: {total_train}")

    def _client_sort_key(k: str):
        try:
            return int(k.split("_")[-1])
        except Exception:
            return 0

    for cid in sorted(federated_train.keys(), key=_client_sort_key):
        print(f"[INFO] {cid}: {len(federated_train[cid])} samples")
    
    val_json = {
        "val_data": val_samples
    }

    save_json(federated_train, args.out, pretty=args.pretty)
    print(f"[OK] Federated train json saved to: {args.out}")

    save_json(val_json, args.val_out, pretty=args.pretty)
    print(f"[OK] Val json saved to: {args.val_out}")

    if args.save_class_map is not None:
        os.makedirs(os.path.dirname(args.save_class_map), exist_ok=True)
        with open(args.save_class_map, "w", encoding="utf-8") as f:
            json.dump(wnid_to_label, f, indent=2, ensure_ascii=False)
        print(f"[OK] Class map saved to: {args.save_class_map}")


if __name__ == "__main__":
    main()