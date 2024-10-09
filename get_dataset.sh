# img2dataset --url_list=myimglist.txt --output_folder=output_folder --thread_count=64 --image_size=256
# img2dataset --url_list=aesthetics_6.5plus.txt --output_folder=aesthetics --thread_count=64 --image_size=256 --output_format "webdataset"
img2dataset --url_list="/workspace/LCM/laion_dataset.parquet" --output_folder=aesthetics --thread_count=64 --image_size=256 --output_format "webdataset" --caption_col "TEXT" --url_cl "URL"