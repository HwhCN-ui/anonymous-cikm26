import argparse
import pprint
import importlib

from sf.core.federated_server import FederatedServer


def get_data_path(data_name,alpha):
    if data_name == "tiny-imagenet":
        federated_train_data_path = "./datasets/data_cache/tiny_imagenet_train.json"
        federated_val_data_path = "./datasets/data_cache/tiny_imagenet_val.json"
    elif data_name == "cifar100":
        if alpha == 0.1:
            federated_train_data_path = "./datasets/data_cache/cifar100_federated_0p1_train.json"
            federated_val_data_path = "./datasets/data_cache/cifar100_federated_0p1_val.json"
        elif alpha == 0.3:
            federated_train_data_path = "./datasets/data_cache/cifar100_federated_0p3_train.json"
            federated_val_data_path = "./datasets/data_cache/cifar100_federated_0p3_val.json"
        elif alpha == 0.9:
            federated_train_data_path = "./datasets/data_cache/cifar100_federated_0p9_train.json"
            federated_val_data_path = "./datasets/data_cache/cifar100_federated_0p9_val.json"
    elif data_name == "cifar10":
        if alpha == 0.1:
            federated_train_data_path = "./datasets/data_cache/cifar10_federated_0p1_train.json"
            federated_val_data_path = "./datasets/data_cache/cifar10_federated_0p1_val.json"
        elif alpha == 0.3:
            federated_train_data_path = "./datasets/data_cache/cifar10_federated_0p3_train.json"
            federated_val_data_path = "./datasets/data_cache/cifar10_federated_0p3_val.json"
        elif alpha == 0.9:
            federated_train_data_path = "./datasets/data_cache/cifar10_federated_0p9_train.json"
            federated_val_data_path = "./datasets/data_cache/cifar10_federated_0p9_val.json"
    else:
        raise ValueError(f"Unsupported dataset name: {data_name!r}")
    return federated_train_data_path, federated_val_data_path


def make_parser():
    parser = argparse.ArgumentParser(description="ViT-LoRA-SF Experiment")

    parser.add_argument("-t", "--data-name", type=str, required=True, help="train federated json path")

    parser.add_argument(
        "-a", "--aggregator",
        type=str.lower,
        choices=["strap", "fedavg", "median",
                 "trimmed",
                 "fltrust", "flshield", "deepsight",
                 "flame", "foolsgold", "lasa", "feddlad"],
        default="strap",
        help="selection aggregation method"
    )

    parser.add_argument(
        "-d", "--attack",
        type=str.lower,
        choices=[
            "none",
            "rank_aware", "alie", "minmax", "minsum", "poisonedfl",
            "signflip", "scaling", "gaussian", "neurotoxin", "cerp", "pfedba"
        ],
        default="none",
        help="update poisoning method"
    )

    parser.add_argument(
        "--backdoor",
        action="store_true",
        default=False,
        help="(legacy) enable backdoor poisoning. If set and --trigger-type=none, defaults to badnets."
    )

    parser.add_argument(
        "--trigger-type",
        type=str.lower,
        choices=["none", "badnets", "tact", "blended"],
        default="none",
        help="Backdoor trigger type. Use none to disable."
    )

    parser.add_argument("-m", "--model-name", type=str.lower, default="vit", help="select model")
    parser.add_argument("-c", "--byzantine-ratio", type=float, default=0.25, help="byzantine ratio")

    parser.add_argument(
        "--strap-aggregator",
        type=str.lower,
        choices=["fedavg", "median", "trimmed", "fltrust",
                 "flshield", "deepsight", "flame", "lasa", "feddlad"],
        default="fedavg",
        help="When --aggregator=strap, choose the base aggregator inside StrAP."
    )

    parser.add_argument(
        "--strap-max-batches",
        type=int,
        default=8,
        help="(Optional) Override the cce_max_batches setting in strap_config.py"
    )
    parser.add_argument("--cce-gamma", type=float, default=None, help="Suppression rate gamma for StrAP (e.g., 3.0)")
    parser.add_argument("--cce-trust-beta", type=float, default=None, help="EMA decay rate beta for StrAP (e.g., 0.85)")
    parser.add_argument("--phi", type=float, default=None, help="Utility calibration threshold phi for StrAP (e.g., 0.7)")

    parser.add_argument(
        "--server-ref-label-mode",
        type=str.lower,
        choices=["all", "keep", "drop", "random_k"],
        default="all",
        help=(
            "How to construct the server reference set used by probe/validation inside StrAP. "
            "'all' = matched; "
            "'keep' = only keep labels in --server-ref-labels; "
            "'drop' = drop labels in --server-ref-labels; "
            "'random_k' = randomly keep K labels."
        )
    )
    parser.add_argument(
        "--server-ref-labels",
        type=str,
        default="",
        help="Comma-separated label ids for mismatch diagnostic, e.g. '0,1,2,3,4'. Used by keep/drop."
    )
    parser.add_argument(
        "--server-ref-random-k",
        type=int,
        default=0,
        help="When --server-ref-label-mode=random_k, keep K labels in the server reference set."
    )
    parser.add_argument(
        "--server-ref-random-seed",
        type=int,
        default=42,
        help="Random seed used by random_k reference-set construction."
    )

    return parser


def _print_configs():
    pp = pprint.PrettyPrinter(indent=2, width=120, sort_dicts=True)

    base = importlib.import_module("sf.config.base_config")
    strap = importlib.import_module("sf.config.strap_config")
    attack = importlib.import_module("sf.config.attack_config")

    print("\n================= CONFIG DUMP (sf.config.*) =================")

    print("\n[base_config]")
    base_keys = ["train", "federated", "log", "tensorboard", "lora", "vit_lora", "data"]
    for k in base_keys:
        if hasattr(base, k) and isinstance(getattr(base, k), dict):
            print(f"\n- {k} =")
            pp.pprint(getattr(base, k))

    print("\n[strap_config]")
    if hasattr(strap, "strap") and isinstance(getattr(strap, "strap"), dict):
        pp.pprint(getattr(strap, "strap"))

    print("\n[attack_config]")
    attack_keys = ["attack", "backdoor", "rank_aware", "alie", "minmax", "minsum",
                   "poisonedfl", "signflip", "scaling", "gaussian", "neurotoxin", "cerp", "pfedba"]
    for k in attack_keys:
        if hasattr(attack, k) and isinstance(getattr(attack, k), dict):
            print(f"\n- {k} =")
            pp.pprint(getattr(attack, k))

    print("=============================================================\n")


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    _print_configs()

    train_data_path, val_data_path = get_data_path(args.data_name,args.alpha)

    trigger_type = (args.trigger_type or "none").lower().strip()
    if args.backdoor and trigger_type == "none":
        trigger_type = "badnets"

    enable_backdoor = (trigger_type != "none")

    print(f"Federated train data path: {train_data_path}")
    print("---------------------------------")
    print(f"Federated val data path: {val_data_path}")
    print("---------------------------------")
    print(f"Aggregator: {args.aggregator}")
    print("---------------------------------")
    print(f"Update Attack: {args.attack}")
    print("---------------------------------")
    print(f"Backdoor Enabled: {enable_backdoor}")
    print("---------------------------------")
    print(f"Trigger Type: {trigger_type}")
    print("---------------------------------")
    print(f"Model: {args.model_name}")
    print("---------------------------------")
    print(f"Byzantine ratio: {args.byzantine_ratio}")
    print("---------------------------------")
    print(f"StrAP Base Aggregator: {args.strap_aggregator}")
    print("---------------------------------")
    print(f"StrAP Max Batches Override: {args.strap_max_batches}")
    print("---------------------------------")
    print(f"CCE Gamma (gamma): {args.cce_gamma}")
    print(f"CCE Trust Beta (beta): {args.cce_trust_beta}")
    print(f"Calibration Threshold (phi): {args.phi}")
    print("---------------------------------")
    print(f"Server Ref Label Mode: {args.server_ref_label_mode}")
    print(f"Server Ref Labels: {args.server_ref_labels}")
    print(f"Server Ref Random K: {args.server_ref_random_k}")
    print(f"Server Ref Random Seed: {args.server_ref_random_seed}")

    print("=== Starting Federated Server ===")
    server = FederatedServer(
        federated_train_data_path=train_data_path,
        federated_val_data_path=val_data_path,
        data_name=args.data_name,
        aggregator=args.aggregator,
        attack=args.attack,
        enable_backdoor=enable_backdoor,
        trigger_type=trigger_type,
        model_name=args.model_name,
        byzantine_ratio=args.byzantine_ratio,
        strap_base_aggregator=args.strap_aggregator,
        strap_max_batches=args.strap_max_batches,
        cce_gamma=args.cce_gamma,
        cce_trust_beta=args.cce_trust_beta,
        phi=args.phi,
        server_ref_label_mode=args.server_ref_label_mode,
        server_ref_labels=args.server_ref_labels,
        server_ref_random_k=args.server_ref_random_k,
        server_ref_random_seed=args.server_ref_random_seed,
    )
    server.run()
