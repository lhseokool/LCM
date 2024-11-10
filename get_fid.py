import torch
from pytorch_fid import fid_score, inception

# paths = ["/workspace/LCM/generated_images/lcm-sdxl", "/workspace/LCM/data/val2017"]
gen_list = ["/workspace/LCM/results/lcm-sdxl_gc1.0",
            "/workspace/LCM/results/lcm-sdxl_gc1.2",
            "/workspace/LCM/results/lcm-sdxl_gc1.4",
            "/workspace/LCM/results/lcm-sdxl_gc1.6",
            "/workspace/LCM/results/lcm-sdxl_gc1.8",
            "/workspace/LCM/results/lcm-sdxl_gc2.0"]

gen_list = ["/workspace/LCM/results/tcd-sdxl-lora_gc00",
            "/workspace/LCM/results/tcd-sdxl-lora_gc01"]


for gen_img_path in gen_list:
    paths = [ "/workspace/LCM/data/val2017", gen_img_path]
    dim = 2048
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fid_value = fid_score.calculate_fid_given_paths(
            paths, batch_size=1, device=device, dims=dim, num_workers=1
    )
    print(gen_img_path)
    print(fid_value)
    print("=" * 100)