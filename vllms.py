from trl.extras.vllm_client import VLLMClient
import json
from transformers import AutoTokenizer
from datasets import load_dataset
import os
import time
import argparse

def main(args):
    test = load_dataset("Leopo1d/OpenVul_Vulnerability_Query_Dataset_for_RL", split=args.mode)

    inputs = {}

    for d in test:
        inputs[d['key']] = {'text': d['prompt']}

    test = None

    # Load the tokenizer corresponding to your model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    
    if not os.path.exists("results"):
        os.makedirs("results", exist_ok=True)
        
    output_path = f"results/{args.name}.json"
    if os.path.exists(output_path):
        with open(output_path, 'r') as f:
            results = json.load(f)
    else:
        results = {}

    all_prompts = [] 
    all_keys = []
    input_keys = [key for key in inputs.keys() if key not in results.keys()]

    count = 0
    batch_size = int(len(input_keys)/2)

    for i in range(int(len(input_keys)/batch_size)):
        group_keys = input_keys[i*batch_size:(i+1)*batch_size]
        ground_prompts = [inputs[k]['text'] for k in group_keys]
        all_prompts.append(ground_prompts)
        all_keys.append(group_keys)
        count += len(group_keys)

    if count != len(input_keys):
        group_keys = input_keys[count:]
        ground_prompts = [inputs[k]['text'] for k in group_keys]
        all_prompts.append(ground_prompts)
        all_keys.append(group_keys)


    server_url = f"http://localhost:8000"
    vllm_client = VLLMClient(base_url=server_url, connection_timeout=10000000)


    for i, ordered_set_of_prompts in enumerate(all_prompts):
        start = time.time()
        try:
            tokenized_prompts = []
            for p in ordered_set_of_prompts: 
                text = tokenizer.apply_chat_template(
                    p,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True
                )
                tokenized_prompts.append(text)
            completion_ids = vllm_client.generate(
                prompts=tokenized_prompts,
                n=8,
                repetition_penalty=1.0,
                temperature=0.6,
                top_p=0.95,
                top_k=20,
                min_p=0,
                max_tokens=32768,
                guided_decoding_regex=None,
                generation_kwargs=None
            )['completion_ids']
            
            group_keys = all_keys[i]
            for j in range(len(tokenized_prompts)):
                prompt_result = completion_ids[j*8:(j+1)*8]
                result = []
                for gen_idx, output in enumerate(prompt_result):
                    decoded_text = tokenizer.decode(
                        output,
                        skip_special_tokens=True
                    )
                    result.append(decoded_text)
                results[group_keys[j]] = result
            
            with open(output_path, 'w') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error: {i}/{len(all_prompts)}: {e}")

        elapsed = time.time() - start

        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)

        print(f"{i}/{len(all_prompts)}:{len(group_keys)} Run time: {minutes} min {seconds} sec")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--name", type=str)
    parser.add_argument("--mode", type=str)
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    main(args)
