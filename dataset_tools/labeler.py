import json
from pathlib import Path

import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

data_path = Path('/media/kwoj/borrowed/Projekt_Uczenie_Maszyn/dnd_scrapped_maps')
dataset = {

}

for image_path in data_path.glob('*.png'):
    raw_image = Image.open(image_path).convert('RGB')

    text = "A top-down tile-based view in a pixel art style depicting a "
    inputs = processor(raw_image, text, return_tensors="pt").to("cuda")
    out = model.generate(**inputs, max_length=100)
    caption = processor.decode(out[0], skip_special_tokens=True)

    dataset[image_path.name] = caption
    print(caption)

    json_file_path = 'dnd_scrapped_maps_dataset.json'

    with open(json_file_path, 'w') as json_file:
        json.dump(dataset, json_file, indent=4)

    print(f"Dictionary saved as JSON in {json_file_path}")
