import os
import json

annotation_dir = './annotations'

json_files = [f for f in os.listdir(annotation_dir) if f.endswith('.json')]

region_data = {}

for json_file in json_files:
    ann_data = json.load(open(os.path.join(annotation_dir, json_file), 'r'))
    # print(ann_data)
    # convert regions
    width, height = ann_data['image_width'], ann_data['image_height']
    image_file_name = ann_data['original_target']
    converted_regions = []
    for region in ann_data['annotations']:
        pos = region['pos']
        converted_regions.append([
            pos['x'] / width,
            pos['y'] / height,
            pos['width'] / width,
            pos['height'] / height,
            0.0,  # not background
            1.0,  # foreground
        ])
    region_data[image_file_name] = converted_regions

with open('cookpad.json', 'w') as savefile:
    json.dump(region_data, savefile)

# save resulting json file