import torch
from pytorch_fid2 import fid_score, inception

paths = ["/workspace/LCM/new/01", "/workspace/LCM/new/02"]
dim = 2048
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fid_value = fid_score.calculate_fid_given_paths(
        paths, batch_size=1, device=device, dims=dim, num_workers=1
    )

print(fid_value)