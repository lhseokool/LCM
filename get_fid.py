import os
import numpy as np
import torch
from PIL import Image
from zipfile import ZipFile
from torchvision.transforms import functional as F
from diffusers import DiTPipeline, DPMSolverMultistepScheduler
from torchmetrics.image.fid import FrechetInceptionDistance

# 데이터 다운로드 함수
def download(url, local_filepath):
    import requests
    r = requests.get(url)
    with open(local_filepath, "wb") as f:
        f.write(r.content)
    return local_filepath

# 이미지 전처리 함수
def preprocess_image(image):
    image = torch.tensor(image).unsqueeze(0)
    image = image.permute(0, 3, 1, 2) / 255.0
    return F.center_crop(image, (256, 256))

# 실제 이미지 로드 함수
def load_real_images(dataset_path):
    image_paths = sorted([os.path.join(dataset_path, x) for x in os.listdir(dataset_path)])
    real_images = [np.array(Image.open(path).convert("RGB")) for path in image_paths]
    real_images = torch.cat([preprocess_image(image) for image in real_images])
    return real_images

# 생성된 이미지 로드 및 처리 함수
def generate_images_from_text(text_labels, model, device="cuda"):
    class_ids = model.get_label_ids(text_labels)
    generator = torch.manual_seed(42)  # 생성 일관성을 위해 시드 고정 (옵션)
    output = model(class_labels=class_ids, generator=generator, output_type="np")
    
    fake_images = torch.tensor(output.images)
    fake_images = fake_images.permute(0, 3, 1, 2)  # [배치, 채널, 높이, 너비]
    return fake_images

# FID 계산 함수
def calculate_fid(real_images, fake_images):
    fid = FrechetInceptionDistance(normalize=True)
    fid.update(real_images, real=True)
    fid.update(fake_images, real=False)
    return float(fid.compute())

# 전체 작업을 수행하는 함수
def calculate_fid_from_text_and_real_images(dataset_url, dataset_path, text_labels):
    # Step 1: 데이터셋 다운로드 및 압축 해제
    local_filepath = download(dataset_url, dataset_url.split("/")[-1])
    with ZipFile(local_filepath, "r") as zipper:
        zipper.extractall(".")
    
    # Step 2: 실제 이미지 로드
    real_images = load_real_images(dataset_path)
    print(f"Real images shape: {real_images.shape}")
    
    # Step 3: DiT 모델 설정 및 이미지 생성
    dit_pipeline = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=torch.float16)
    dit_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(dit_pipeline.scheduler.config)
    dit_pipeline = dit_pipeline.to("cuda")
    
    # Step 4: 텍스트 기반 이미지 생성
    fake_images = generate_images_from_text(text_labels, dit_pipeline)
    print(f"Fake images shape: {fake_images.shape}")
    
    # Step 5: FID 계산
    fid_value = calculate_fid(real_images, fake_images)
    print(f"FID: {fid_value}")
    return fid_value



if __name__ == "__main__":
    dummy_dataset_url = "https://hf.co/datasets/sayakpaul/sample-datasets/resolve/main/sample-imagenet-images.zip"
    dataset_path = "sample-imagenet-images"
    text_labels = [
        "cassette player",
        "chainsaw",
        "chainsaw",
        "church",
        "gas pump",
        "gas pump",
        "gas pump",
        "parachute",
        "parachute",
        "tench",
    ]

    fid_value = calculate_fid_from_text_and_real_images(dummy_dataset_url, dataset_path, text_labels)
    print(f"Final FID Value: {fid_value}")