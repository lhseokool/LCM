import torch
import numpy as np
from diffusers import StableDiffusionPipeline
from torchmetrics.functional.multimodal import clip_score
from functools import partial

# Stable Diffusion 모델 로드 및 설정
def load_sd_pipeline(model_ckpt, device="cuda"):
    pipeline = StableDiffusionPipeline.from_pretrained(model_ckpt, torch_dtype=torch.float16).to(device)
    return pipeline

# 이미지 생성 함수
def generate_images_from_prompts(pipeline, prompts, num_images_per_prompt=1, seed=None):
    if seed is not None:
        generator = torch.manual_seed(seed)
    else:
        generator = None
    
    images = pipeline(prompts, num_images_per_prompt=num_images_per_prompt, generator=generator, output_type="np").images
    return images

# CLIP 스코어 계산 함수
def calculate_clip_score(images, prompts, model_name_or_path="openai/clip-vit-base-patch16"):
    clip_score_fn = partial(clip_score, model_name_or_path=model_name_or_path)
    
    # 이미지를 0~1 범위로 변환하고 정수로 변환
    images_int = (images * 255).astype("uint8")
    
    # 이미지 차원을 변환하여 CLIP 모델 입력에 맞춤
    clip_score_value = clip_score_fn(torch.from_numpy(images_int).permute(0, 3, 1, 2), prompts).detach()
    return round(float(clip_score_value), 4)

# 전체 파이프라인을 실행하는 함수
def calculate_clip_score_from_texts_and_images(model_ckpt, prompts, num_images_per_prompt=1, seed=None):
    # Step 1: Stable Diffusion 모델 로드
    sd_pipeline = load_sd_pipeline(model_ckpt)
    
    # Step 2: 텍스트 기반 이미지 생성
    images = generate_images_from_prompts(sd_pipeline, prompts, num_images_per_prompt=num_images_per_prompt, seed=seed)
    
    # Step 3: CLIP 스코어 계산
    clip_score_value = calculate_clip_score(images, prompts)
    return clip_score_value

# 사용 예시
if __name__ == "__main__":
    model_ckpt = "CompVis/stable-diffusion-v1-4"
    prompts = [
        "a photo of an astronaut riding a horse on mars",
        "A high tech solarpunk utopia in the Amazon rainforest",
        "A pikachu fine dining with a view to the Eiffel Tower",
        "A mecha robot in a favela in expressionist style",
        "an insect robot preparing a delicious meal",
        "A small cabin on top of a snowy mountain in the style of Disney, artstation",
    ]
    
    # CLIP 스코어 계산
    clip_score_value = calculate_clip_score_from_texts_and_images(model_ckpt, prompts, num_images_per_prompt=1, seed=42)
    print(f"CLIP Score: {clip_score_value}")