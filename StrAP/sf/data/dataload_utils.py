from sf.config.base_config import data as data_cfg
from .tiny_imagenet_dataset import TinyImageNetFederated
from .cifar100_dataset import CIFARF100Federated

def get_dataset(
    federated_data_path: str,
    model_name: str,
    client_id=0,
    data_name=None,
    is_server: bool = False,
):
    
    if data_name is None:
        raise ValueError("data_name must be provided")
    
    name = (data_name).lower()
    print(f"Dataset name: {name}")
    if name in ["tiny_imagenet", "tiny-imagenet", "tiny","cifar10"]:
        return TinyImageNetFederated(
            federated_data_path=federated_data_path,
            model_name=model_name,
            client_id=client_id,
            is_server=is_server
        )
    elif name in ["cifar100"]:
        print(f"federated_data_path: {federated_data_path}")
        return CIFARF100Federated(
            federated_data_path=federated_data_path,
            model_name=model_name,
            client_id=client_id,
            is_server=is_server
        )

    raise ValueError(f"Unknown dataset name: {name!r}")
