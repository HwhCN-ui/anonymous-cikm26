import os
import json
import argparse
import pickle
import numpy as np
from typing import Dict, List, Tuple

from PIL import Image


def find_cifar_python_root(root: str) -> str:
    root = os.path.abspath(root)

    candidates = [
        root,
        os.path.join(root, "cifar-100-python"),
        os.path.join(root, "data"),
        os.path.join(root, "data", "cifar-100"),
        os.path.join(root, "data", "cifar-100", "cifar-100-python"),
    ]

    print(f"[INFO] Searching for cifar-100-python in root: {root}")
    for c in candidates:
        print(f"[INFO] Checking candidate path: {c}")
        if os.path.basename(c.rstrip(os.sep)) == "cifar-100-python" and \
           os.path.isfile(os.path.join(c, "train")) and \
           os.path.isfile(os.path.join(c, "test")):
            return c

        inner = os.path.join(c, "cifar-100-python")
        if os.path.isdir(inner) and \
           os.path.isfile(os.path.join(inner, "train")) and \
           os.path.isfile(os.path.join(inner, "test")):
            return inner

    raise FileNotFoundError(
        f"Cannot find 'cifar-100-python' (with train/test/meta) under root={root!r}."
    )


def load_cifar100_split(python_root: str, split: str) -> Tuple[np.ndarray, List[int], List[str]]:
    assert split in ["train", "test"]
    fpath = os.path.join(python_root, split)
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"{split!r} file not found in {python_root!r}")

    with open(fpath, "rb") as f:
        entry = pickle.load(f, encoding="latin1")

    data = entry["data"]
    labels = entry["fine_labels"]
    filenames = entry.get("filenames", [f"{split}_{i}" for i in range(len(labels))])

    data = data.reshape(-1, 3, 32, 32)
    data = data.astype("uint8")

    return data, labels, filenames


def load_class_map(python_root: str) -> Dict[str, int]:
    meta_path = os.path.join(python_root, "meta")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"meta file not found in {python_root!r}")

    with open(meta_path, "rb") as f:
        meta = pickle.load(f, encoding="latin1")

    fine_label_names = meta["fine_label_names"]
    class_to_id = {name: idx for idx, name in enumerate(fine_label_names)}
    return class_to_id


def build_image_samples(
    images: np.ndarray,
    labels: List[int],
    split_name: str,
    img_root: str,
    label_key: str,
    is_val: bool,
) -> List[dict]:
    os.makedirs(img_root, exist_ok=True)
    split_root = os.path.join(img_root, split_name)
    os.makedirs(split_root, exist_ok=True)

    samples: List[dict] = []

    num = images.shape[0]
    for idx in range(num):
        img_arr = images[idx]
        label = int(labels[idx])

        label_dir = os.path.join(split_root, f"{label}")
        os.makedirs(label_dir, exist_ok=True)

        img_arr = np.transpose(img_arr, (1, 2, 0))
        img = Image.fromarray(img_arr)

        filename = f"{split_name}_{idx:05d}.png"
        img_path = os.path.abspath(os.path.join(label_dir, filename))
        img.save(img_path)

        if is_val:
            sample = {
                "img_path": img_path,
                "label": label,
            }
        else:
            sample = {
                "img_path": img_path,
                label_key: label,
            }

        samples.append(sample)

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
    ap = argparse.ArgumentParser(description="Convert CIFAR-100 (cifar-100-python) to federated classification JSON.")
    ap.add_argument("--root", required=True, help="Root directory containing cifar-100-python/ (can have data/cifar-100 outside)")
    ap.add_argument("--alpha", type=float, default=0.3, help="Dirichlet alpha value (default: 0.3)")
    ap.add_argument("--out", default="./datasets/data_cache/cifar100_train.json",
                    help="Federated training JSON output path")
    ap.add_argument("--val_out", default="./datasets/data_cache/cifar100_val.json",
                    help="Validation JSON output path")
    ap.add_argument("--num_clients", type=int, default=20, help="Number of federated clients")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pretty", action="store_true", help="Save JSON with indented formatting")
    ap.add_argument("--label-key", default="label",
                    help="Label field name in training JSON (pass --label-key lable if compatibility with 'lable' is needed)")
    ap.add_argument("--save_class_map", default=None,
                    help="Optional: Path to save class_name->id mapping JSON")

    args = ap.parse_args()

    np.random.seed(args.seed)

    cifar_py_root = find_cifar_python_root(args.root)
    print(f"[INFO] cifar-100-python root: {cifar_py_root}")

    img_root = os.path.join(cifar_py_root, "images")

    class_map = load_class_map(cifar_py_root)
    print(f"[INFO] #classes: {len(class_map)}")

    train_imgs, train_labels, _ = load_cifar100_split(cifar_py_root, "train")
    test_imgs, test_labels, _ = load_cifar100_split(cifar_py_root, "test")

    print(f"[INFO] #raw train samples: {train_imgs.shape[0]}")
    print(f"[INFO] #raw test  samples: {test_imgs.shape[0]}")

    print(f"[INFO] Saving train images to {img_root} ...")
    train_samples = build_image_samples(
        images=train_imgs,
        labels=train_labels,
        split_name="train",
        img_root=img_root,
        label_key=args.label_key,
        is_val=False,
    )

    print(f"[INFO] Saving val/test images to {img_root} ...")
    val_samples = build_image_samples(
        images=test_imgs,
        labels=test_labels,
        split_name="val", 
        img_root=img_root,
        label_key=args.label_key,
        is_val=True,       
    )

    print(f"[INFO] #train samples (after save): {len(train_samples)}")
    print(f"[INFO] #val   samples (after save): {len(val_samples)}")

    print(f"[INFO] Dirichlet alpha = {args.alpha}, num_clients = {args.num_clients}")
    federated_train = dirichlet_split_non_iid(
        samples=train_samples,
        label_key=args.label_key,
        num_clients=args.num_clients,
        alpha=args.alpha,
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

    alpha_tag = str(args.alpha).replace(".", "p")
    federated_train_data_path = f"./datasets/data_cache/cifar100_federated_{alpha_tag}_train.json"
    federated_val_data_path = f"./datasets/data_cache/cifar100_federated_{alpha_tag}_val.json"
    
    save_json(federated_train, federated_train_data_path, pretty=args.pretty)
    print(f"[OK] Federated train json saved to: {federated_train_data_path}")

    save_json(val_json, federated_val_data_path, pretty=args.pretty)
    print(f"[OK] Val json saved to: {federated_val_data_path}")

    if args.save_class_map is not None:
        os.makedirs(os.path.dirname(args.save_class_map), exist_ok=True)
        with open(args.save_class_map, "w", encoding="utf-8") as f:
            json.dump(class_map, f, indent=2, ensure_ascii=False)
        print(f"[OK] Class map saved to: {args.save_class_map}")


if __name__ == "__main__":
    main()