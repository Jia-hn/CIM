import os
import datasets
import argparse
from PIL import Image
from io import BytesIO


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", type=str, default="data/SC-Captioner-data/llamafactory_json", help="The save directory for the preprocessed dataset.")
    parser.add_argument("--dataset_name", type=str, default="train_coco6k_sft", help="The name of the dataset.")
    args = parser.parse_args()
    local_dir = args.local_dir
    dataset_name = args.dataset_name
    data_source = os.path.join(local_dir, f"{dataset_name}.json")
    dataset = datasets.load_dataset("json", data_files=data_source)
    dataset = dataset['train']

    def make_map_fn():
        def process_fn(example, index):
            prompt = example["conversations"][0]["value"]
            captions = example["conversations"][1]["value"]
            with open(example["images"][0], "rb") as f:
                image_bytes = f.read()
            image = Image.open(BytesIO(image_bytes))
            data = {
                "data_source": dataset_name,
                "prompt": [
                    {
                        "role": "system",
                        "content": "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "images": [image],
                "captions": captions,
                "extra_info": {
                    "index": index,
                },
            }
            return data

        return process_fn

    dataset = dataset.map(function=make_map_fn(), with_indices=True, num_proc=16)

    dataset.to_parquet(os.path.join(local_dir, f"{dataset_name}_llava.parquet"))
