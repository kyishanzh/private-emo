import argparse
import torch

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir",    type=str, required=True)
    p.add_argument("--batch-size",  type=int, default=32)
    p.add_argument("--epochs",      type=int, default=5)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--device",      type=str, default="cpu",
                   help="cpu or cuda")
    p.add_argument("--max-samples", type=int, default=None,
                   help="for quick CPU smoke test")
    return p.parse_args()

def get_device(device_str):
    return torch.device(device_str if torch.cuda.is_available() or device_str=="cpu"
                       else "cpu")
