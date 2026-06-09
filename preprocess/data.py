import os
import datasets
import argparse


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
            data = {
                "data_source": dataset_name,
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "images": example["images"][0],
                "captions": captions,
                "extra_info": {
                    "index": index,
                },
            }
            return data

        return process_fn

    dataset = dataset.map(function=make_map_fn(), with_indices=True, num_proc=16)

    dataset.to_parquet(os.path.join(local_dir, f"{dataset_name}.parquet"))
