# python -m pytorch_fid --save-stats /workspace/LCM/generated_image.jpg /workspace/LCM/save

# python -m pytorch_fid /workspace/LCM/generated_image.jpg /workspace/LCM/real_image.jpg

python dataset_tool.py --source /workspace/LCM/data/annotations/captions_val2017.json \
  --dest data/coco_val256.zip --resolution 256x256 --transform center-crop