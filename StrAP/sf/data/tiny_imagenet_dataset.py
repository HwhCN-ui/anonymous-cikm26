
import json
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from typing import List, Dict, Any, Union, Optional

from sf.config.base_config import data as data_cfg, federated as fl_cfg


class _FederatedJsonImageDataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]], image_size: int):
        self.samples = samples
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        if "img_path" not in s:
            raise KeyError("Sample missing required key: 'img_path'.")
        if "label" not in s:
            raise KeyError("Sample missing required key: 'label'.")

        img = Image.open(s["img_path"]).convert("RGB")
        label = int(s["label"])
        x = self.transform(img)

        return {"pixel_values": x, "labels": label}


class TinyImageNetFederated:
    """
    训练集：
        {
          "client_1": [ {"img_path":..., "label":...}, ... ],
          ...
          "client_N": [...]
        }

    验证集：
        {
          "val_data": [ {"img_path":..., "label":...}, ... ]
        }

    规则：
    - is_server=True：如果是 val json -> 读 val_data
                       如果误传了 train json -> 合并所有 client_*（方便调试）
    - is_server=False：必须能找到对应 client_k
    """
    def __init__(
        self,
        federated_data_path: str,
        model_name: str,
        client_id: Union[int, str] = 0,
        is_server: bool = False,
    ):
        self.federated_data_path = federated_data_path
        self.model_name = model_name
        self.client_id = client_id
        self.is_server = is_server
        self.image_size = int(data_cfg.get("image_size", 224))

        with open(federated_data_path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        self.samples = self._parse_samples(obj)

    def _normalize_client_key(self, client_id: Union[int, str]) -> str:
        if isinstance(client_id, str):
            if client_id.startswith("client_"):
                return client_id
            if client_id.isdigit():
                return f"client_{int(client_id)}"
            raise ValueError(f"Invalid client_id string: {client_id!r}. Expected 'client_k' or digit.")

        return f"client_{int(client_id)}"

    def _parse_samples(self, obj: Dict[str, Any]) -> List[Dict[str, Any]]:
        if "val_data" in obj:
            if not isinstance(obj["val_data"], list):
                raise TypeError(f"'val_data' must be a list, got {type(obj['val_data'])}.")
            return obj["val_data"]

        client_keys = [k for k in obj.keys() if isinstance(k, str) and k.startswith("client_")]
        if len(client_keys) == 0:
            raise ValueError(
                "Unrecognized json format. "
                "Train json must contain keys like 'client_1'...'client_N', "
                "or val json must contain key 'val_data'."
            )

        if self.is_server:
            merged = []
            for k in sorted(client_keys):
                v = obj[k]
                if not isinstance(v, list):
                    raise TypeError(f"Train json key {k!r} must be a list, got {type(v)}.")
                merged.extend(v)
            return merged

        ck = self._normalize_client_key(self.client_id)
        if ck not in obj:
            raise KeyError(
                f"Client key {ck!r} not found in train json. "
                f"Available client keys: {sorted(client_keys)}"
            )
        if not isinstance(obj[ck], list):
            raise TypeError(f"Train json key {ck!r} must be a list, got {type(obj[ck])}.")

        return obj[ck]

    def get_loader(self, batch_size: Optional[int] = None, shuffle: bool = True):
        batch_size = int(batch_size or fl_cfg["batch_size"])
        ds = _FederatedJsonImageDataset(self.samples, self.image_size)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=fl_cfg["num_workers"],
            prefetch_factor=fl_cfg["prefetch_factor"],
            pin_memory=True
        )
