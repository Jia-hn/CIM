import argparse
import nltk
nltk.data.path.insert(0, "nltk_data")
from PIL import Image
from transformers import AutoTokenizer
import json
from vllm import LLM, SamplingParams
import os
from pathlib import Path
import metric.capture_sc
import metric.capture
import pickle
import re

def _example_by_n(n: int) -> str:
    if n == 3:
        return '1: Yes, 2: No, 3: n/a'
    if n == 4:
        return '1: Yes, 2: No, 3: Yes, 4: n/a'
    return '1: Yes, 2: No, 3: Yes, 4: n/a, 5: Yes'

def _build_vqa_prompt(caption: str, questions: list[str]) -> str:
    n = len(questions)
    q_lines = [f"{i}. {q}" for i, q in enumerate(questions, 1)]
    q_block = "".join(q_lines)
    example = _example_by_n(n)
    return (
        f'I will give you a passage of caption. Please answer the following {n} questions with "Yes", "No", or "n/a" based on the given caption. '
        f'Output like this: "{example}". Don\'t output extra text.\n'
        f'Caption:"{caption}"\n'
        f'Questions:{q_block}'
    )

def _parse_vqa_output(text: str, n):
    pat = re.compile(r'(\b(\d+)\s*[:\-]?\s*)(yes|no|n/?a)\b', re.IGNORECASE)
    slots = [None] * n
    for m in pat.finditer(text):
        idx = int(m.group(2)) - 1
        if 0 <= idx < n:
            slots[idx] = m.group(3)
    return slots

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, default="rl", choices=["rl", "sft", "base"])
    parser.add_argument("--project_name", type=str, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--save_freq", type=int, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    args = parser.parse_args()
    project_name = args.project_name
    experiment_name = args.experiment_name
    dataset_names = ["cocoln500", "docci500"]

    if args.model_type == "base":
        assert args.model_path is not None, "--model_path is required for base models"
        model_path = args.model_path
        eval_subdir = os.path.basename(model_path.rstrip("/"))
    else:
        assert project_name is not None, "--project_name is required for rl/sft models"
        assert args.save_freq is not None, "--save_freq is required for rl/sft models"
        step = (int(os.environ.get("RANK")) + 1) * args.save_freq
        if args.model_type == "rl":
            assert experiment_name is not None, "--experiment_name is required for rl models"
            model_path = f"checkpoints/{project_name}/{experiment_name}/global_step_{step}/actor/huggingface"
            eval_subdir = f"{project_name}/{experiment_name}/checkpoint-{step}"
        else:
            model_path = f"checkpoints/{project_name}/checkpoint-{step}"
            eval_subdir = f"{project_name}/checkpoint-{step}"
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        dtype="auto",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
    )
    sampling = SamplingParams(
        max_tokens=1024,
        temperature=0.0,
    )

    for dataset_name in dataset_names:
        data_path = f"data/SC-Captioner-data/llamafactory_json/test_{dataset_name}.json"
        pred_path = f"evaluation/{eval_subdir}/{dataset_name}_cand.json"
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        requests = []
        for item in data:
            img_path = item["images"][0]
            messages = [{
                "role": "user",
                "content": "<image>Caption this image as accurately as possible, without speculation. Describe what you see.",
            }]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            pil_img = Image.open(img_path).convert("RGB")
            requests.append({
                "prompt": prompt,
                "multi_modal_data": {"image": pil_img}
            })

        outputs = llm.generate(requests, sampling_params=sampling, use_tqdm=True)

        result_dict = {}
        for i, out in enumerate(outputs):
            result_dict[os.path.splitext(os.path.basename(data[i]["images"][0]))[0]] = out.outputs[0].text

        Path(pred_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(pred_path, "w", encoding="utf-8") as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=4)

        pred_path = f"evaluation/{eval_subdir}/{dataset_name}_cand.json"
        gt_path = f"evaluation/{dataset_name}_gt.json"
        cand_parsed_path = f"evaluation/{eval_subdir}/{dataset_name}_cand.pkl"
        gt_parsed_path = f"evaluation/{dataset_name}_gt.pkl"
        with open(gt_path, 'r', encoding='utf-8') as file:
            gt_dict = json.load(file)
        with open(pred_path, 'r', encoding='utf-8') as file:
            pred_dict = json.load(file)

        refs = {}
        preds = {}

        for image_name in pred_dict:
            refs[image_name] = [gt_dict[image_name]]
            preds[image_name] = [pred_dict[image_name]]
        
        extra_objects = {}
        with open(f"evaluation/{dataset_name}_gpt_objects.jsonl", "r") as f:
            for line in f:
                item = json.loads(line)
                key = next(iter(item))
                extra_objects[key] = item[key]
        with open(f"evaluation/{dataset_name}_gpt_attributes.json", "r") as f:
            extra_attributes = json.load(f)

        if os.path.isfile(gt_parsed_path):
            with open(gt_parsed_path, 'rb') as file: 
                prev_gt_parsed = pickle.load(file)
        else:
            prev_gt_parsed = None

        if os.path.isfile(cand_parsed_path):
            with open(cand_parsed_path, 'rb') as file: 
                prev_cand_parsed = pickle.load(file)
        else:
            prev_cand_parsed = None
            
        object_precision, object_recall, object_f1, attribute_precision, attribute_recall, attribute_f1, cand_parsed, gt_parsed = metric.capture_sc.CAPTURE().compute_score(refs, preds, prev_gt_parsed=prev_gt_parsed, prev_cand_parsed=prev_cand_parsed, extra_objects=extra_objects, extra_attributes=extra_attributes, return_parse_results=True)
        score, scores = metric.capture.CAPTURE().compute_score(refs, preds, prev_gt_parsed=gt_parsed, prev_cand_parsed=cand_parsed)
        with open(f"evaluation/{eval_subdir}/{dataset_name}_metrics.txt", "a") as f: 
            print(f"CAPTURE score: {score*100:.2f}", file=f)
            print(f"object precision: {object_precision*100:.2f}", file=f)
            print(f"object recall: {object_recall*100:.2f}", file=f)
            print(f"object f1: {object_f1*100:.2f}", file=f)
            print(f"attribute precision: {attribute_precision*100:.2f}", file=f)
            print(f"attribute recall: {attribute_recall*100:.2f}", file=f)
            print(f"attribute f1: {attribute_f1*100:.2f}", file=f)
        
        with open(cand_parsed_path, 'wb') as file: 
            pickle.dump(cand_parsed, file)

        with open(gt_parsed_path, 'wb') as file: 
            pickle.dump(gt_parsed, file)

    try:
        llm.terminate()
    except Exception:
        pass
    del llm, tokenizer

    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()

    vqa_model_path = 'Qwen/Qwen3-8B'
    vqa_tokenizer = AutoTokenizer.from_pretrained(vqa_model_path, trust_remote_code=True)
    vqa_llm = LLM(
        model=vqa_model_path,
        trust_remote_code=True,
        dtype="auto",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
    )
    vqa_sampling = SamplingParams(
        max_tokens=128,
        temperature=0.0,
    )

    for dataset_name in dataset_names:
        pred_path = f"evaluation/{eval_subdir}/{dataset_name}_cand.json"
        with open(pred_path, 'r', encoding='utf-8') as file:
            pred_dict = json.load(file)
        vqa_data_path = f'evaluation/{dataset_name}_questions.json'
        with open(vqa_data_path, "r", encoding="utf-8") as f:
            vqa_data = json.load(f)

        vqa_requests = []
        vqa_index = []
        for image_name, caption in pred_dict.items():
            if image_name not in vqa_data:
                continue
            qa_dict = vqa_data[image_name]
            q_list = list(qa_dict.keys())

            gt_list = [qa_dict[q] for q in q_list]

            user_prompt = _build_vqa_prompt(caption, q_list)
            messages = [{"role": "user", "content": user_prompt}]
            prompt = vqa_tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True,
                enable_thinking=False
            )
            vqa_requests.append({"prompt": prompt})
            vqa_index.append((image_name, q_list, gt_list))

        vqa_outputs = vqa_llm.generate(vqa_requests, sampling_params=vqa_sampling, use_tqdm=True)

        total, correct = 0, 0
        vqa_pred_detail = {}
        for i, out in enumerate(vqa_outputs):
            pred_str = out.outputs[0].text
            image_name, q_list, gt_list = vqa_index[i]
            n = len(q_list)
            pred_ans = _parse_vqa_output(pred_str, n)

            per_image_pred = {}
            for k in range(n):
                pa = pred_ans[k]
                ga = gt_list[k]
                per_image_pred[q_list[k]] = pa
                total += 1
                if pa == ga:
                    correct += 1
            vqa_pred_detail[image_name] = per_image_pred

        vqa_acc = (correct / total) if total > 0 else 0.0

        pred_out_path = f"evaluation/{eval_subdir}/{dataset_name}_vqa_pred.json"
        with open(pred_out_path, "w", encoding="utf-8") as f:
            json.dump(vqa_pred_detail, f, ensure_ascii=False, indent=2)

        with open(f"evaluation/{eval_subdir}/{dataset_name}_metrics.txt", "a") as f:
            print(f"VQA accuracy: {vqa_acc*100:.2f}", file=f)
    try:
        vqa_llm.terminate()
    except Exception:
        pass
    del vqa_llm, vqa_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

if __name__=='__main__':
    main()
