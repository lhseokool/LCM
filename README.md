# LCM
## Download COCO 2017 Val Images

Download the COCO 2017 validation images (5,000 images, approximately 1GB in size):

```bash
wget http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip
```

After downloading and extracting the files, you should have the following structure:
```bash
val2017/                          # Directory containing the validation images
annotations/captions_val2017.json # Caption annotations for validation images
```

## Evaluations
### FID
- [pytorch-fid](https://github.com/mseitzer/pytorch-fid)
- `get_fid.py`
### Image Complexity
- [IC9600](https://github.com/tinglyfeng/IC9600)
- `get_complexity.py`
### Image Reward
- [ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation](https://github.com/THUDM/ImageReward)
- `get_reward.py`
### PickScore
- [Pick-a-Pic: An Open Dataset of User Preferences for Text-to-Image Generation](https://github.com/yuvalkirstain/PickScore)
- `get_pickscore.py`