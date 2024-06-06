import json

with open('dnd_scrapped_maps_dataset.json', 'r') as f:
    data_dict = json.load(f)

transformed_data = []
for idx, (file_name, text) in enumerate(data_dict.items()):
    new_entry = {
        "file_name": file_name,
        "text": text
    }
    transformed_data.append(new_entry)

with open('metadata.jsonl', 'w') as f:
    for entry in transformed_data:
        json.dump(entry, f)
        f.write('\n')

print("Data transformation complete. Saved to 'transformed_metadata.jsonl'")
