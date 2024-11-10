# python -m pytorch_fid --save-stats /workspace/LCM/generated_image.jpg /workspace/LCM/save

# python -m pytorch_fid /workspace/LCM/generated_image.jpg /workspace/LCM/real_image.jpg

# python dataset_tool.py --source /workspace/LCM/data/annotations/captions_val2017.json \
#   --dest data/coco_val256.zip --resolution 256x256 --transform center-crop


# python calc_metrics.py --metrics fid50k_full --network PATH_TO_NETWORK_PKL


python IC9600/gene.py --input /workspace/LCM/generated_images/tcd-sdxl-lora --output complexity/tcd-sdxl-lora