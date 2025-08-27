import json

data_path = "datasets/ytvis_2019/train.json"
new_data_path = "datasets/ytvis_2019/train_new.json"

with open(data_path, "r") as f:
    data = json.load(f)

# change all category_id to 1 (class-agnostic)
for anno in data["annotations"]:
    anno["category_id"] = 1

# reduce the number of the categories to 1 (class-agnostic)
data["categories"] = [
    {
        "supercategory": "object",
        "id": 1,
        "name": "object",
    },
]

# save the data
with open(new_data_path, "w") as f:
    json.dump(data, f)