import os
import torch
import numpy as np
from diffusers import UNet2DConditionModel, DiffusionPipeline, LCMScheduler
from pycocotools.coco import COCO
from torchvision.models import inception_v3
from torchvision import transforms
from scipy.linalg import sqrtm
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance

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

# 2. InceptionV3에서 feature 추출
def get_inception_features(image):
    model = inception_v3(pretrained=True, transform_input=False)
    model.fc = torch.nn.Identity()  # Feature extractor
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        features = model(image)
    
    return features.cpu().numpy().squeeze()

# 3. FID 계산 함수
def calculate_fid(real_features, generated_features):
    mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu_gen, sigma_gen = np.mean(generated_features, axis=0), np.cov(generated_features, rowvar=False)
    
    diff = mu_real - mu_gen
    cov_mean, _ = sqrtm(sigma_real @ sigma_gen, disp=False)
    
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real
    
    fid = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2 * cov_mean)
    return fid

# 4. 이미지 생성 및 FID 계산
def calculate_fid_for_first_image(image_dir, annotation_file):
    # COCO 데이터셋에서 첫 번째 이미지와 캡션 로드
    img_caption_mapping = load_coco_annotations(annotation_file)
    first_image_file = list(img_caption_mapping.keys())[0]
    first_image_path = os.path.join(image_dir, first_image_file)
    real_image = Image.open(first_image_path).convert('RGB')
    
    # Diffusion 모델로 이미지 생성
    unet = UNet2DConditionModel.from_pretrained("latent-consistency/lcm-sdxl", torch_dtype=torch.float16, variant="fp16")
    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", unet=unet, torch_dtype=torch.float16, variant="fp16")
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    
    # # 캡션을 이용하여 이미지 생성
    prompt = img_caption_mapping[first_image_file][0]  # 첫 번째 이미지의 첫 번째 캡션 사용
    generated_image = pipe(prompt, num_inference_steps=4, guidance_scale=8.0).images[0]
    
    generated_image.save("generated_image.jpg")
    real_image.save("real_image.jpg")

    # # 실제 이미지와 생성된 이미지에서 InceptionV3 feature 추출
    # real_features = get_inception_features(real_image)  # 2048
    # generated_features = get_inception_features(generated_image)  # 2048
    
    # fid = FrechetInceptionDistance(normalize=True)
    # fid.update(real_features, real=True)
    # fid.update(generated_features, real=False)

    # print(f"FID: {float(fid.compute())}")
        
    # # print(real_features.shape)
    # # print(generated_features.shape)
    # # # FID 계산
    # # fid_score = calculate_fid(np.array([real_features]), np.array([generated_features]))
    
    # return fid_score

# COCO annotations 파일 경로와 이미지 디렉토리 설정
annotation_file = "/workspace/LCM/data/annotations/captions_val2017.json"
image_dir = "/workspace/LCM/data/val2017"

# 첫 번째 이미지의 FID 계산
fid_score = calculate_fid_for_first_image(image_dir, annotation_file)
print(f"FID Score: {fid_score}")