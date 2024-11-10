import os
import torch
import random
from diffusers import UNet2DConditionModel, DiffusionPipeline, LCMScheduler
from pycocotools.coco import COCO
from PIL import Image


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    # GPU에서의 시드 고정
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

# 1. COCO annotations 파일 로드
def load_coco_annotations(annotation_file):
    coco = COCO(annotation_file)
    img_ids = coco.getImgIds()
    img_caption_mapping = {}

    for img_id in img_ids:
        img_info = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        img_file = img_info['file_name']
        captions = [ann['caption'] for ann in anns]
        img_caption_mapping[img_file] = captions
    
    return img_caption_mapping

# 2. 이미지 생성 함수
def generate_images_with_random_caption(annotation_file, model_name, guidance_scale=1.0):
    # COCO 데이터셋 로드
    img_caption_mapping = load_coco_annotations(annotation_file)

    # 출력 폴더가 없으면 생성
    output_dir = f"/workspace/LCM/results/{model_name}_gc{guidance_scale:02}"
    os.makedirs(output_dir, exist_ok=True)

    # Diffusion 모델 초기화
    if model_name == "sdxl":
        from diffusers import  StableDiffusionXLPipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        )
        
    elif model_name == "lcm-sdxl":
        unet = UNet2DConditionModel.from_pretrained("latent-consistency/lcm-sdxl", torch_dtype=torch.float16, variant="fp16")
        pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", unet=unet, torch_dtype=torch.float16, variant="fp16")
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
        
    elif model_name == "tcd-sdxl-lora":
        from diffusers import  StableDiffusionXLPipeline
        from scheduling_tcd import TCDScheduler 
        base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        tcd_lora_id = "h1t/TCD-SDXL-LoRA"

        pipe = StableDiffusionXLPipeline.from_pretrained(base_model_id, torch_dtype=torch.float16, variant="fp16")
        pipe.scheduler = TCDScheduler.from_config(pipe.scheduler.config)

        pipe.load_lora_weights(tcd_lora_id)
        pipe.fuse_lora()
        
    pipe.to("cuda")

    # 각 이미지에 대해 랜덤 캡션을 사용하여 이미지 생성 및 저장
    num_image = 0
    for img_file, captions in img_caption_mapping.items():
        # 랜덤 캡션 선택p
        # prompt = random.choice(captions)
        prompt = captions[0]
        
        # 이미지 생성
        if model_name == "sdxl":
            generated_image = pipe(prompt, guidance_scale=guidance_scale).images[0]
        elif model_name == "lcm-sdxl":
            generated_image = pipe(prompt, num_inference_steps=4, guidance_scale=guidance_scale).images[0]
        elif model_name == "tcd-sdxl-lora":
            generated_image = pipe(prompt,num_inference_steps=4, guidance_scale=guidance_scale, eta=0.3,
                                   generator=torch.Generator(device="cuda").manual_seed(0),).images[0]
        # 생성된 이미지 저장
        output_path = os.path.join(output_dir, f"generated_{num_image+1}.jpg")
        generated_image.save(output_path)
        print(f"Generated image saved at: {output_path}")
        num_image += 1

# COCO annotations 파일 경로, 이미지 디렉토리, 출력 폴더 설정
set_seed(42)
annotation_file = "/workspace/LCM/data/annotations/captions_val2017.json"
model_name = "sdxl"
# model_name = "lcm-sdxl"
# model_name = "tcd-sdxl-lora"

guidance_scale_list = [4, 6, 8]
# 이미지 생성 및 저장

for guidance_scale in guidance_scale_list:
    generate_images_with_random_caption(annotation_file, model_name, guidance_scale)