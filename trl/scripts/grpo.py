# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# /// script
# dependencies = [
#     "trl",
#     "peft",
#     "trackio",
#     "kernels",
# ]
# ///

import argparse
from dataclasses import dataclass, field
from typing import Optional

from accelerate import logging
from datasets import load_dataset

# from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.distributed as dist
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
import json
import asyncio
from grpo_utils import AsyncOpenAIAnalyzer, construct_reasoning_level_judge_prompt, construct_specification_level_judge_prompt, json_parser_reasoning, json_parser_specification
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["TRITON_CACHE_DIR"] = "/projects/ksun3/xwang44/xdg_cache/.triton/autotune"
from datetime import timedelta
dist.init_process_group(backend='nccl', init_method='env://', timeout=timedelta(hours=24))
import time
from openai import AsyncOpenAI
import torch
torch._inductor.config.autotune_local_cache = False
import re
logger = logging.get_logger(__name__)

# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")

cwe_1000 = load_dataset("Leopo1d/OpenVul_CWE_Hierarchical_Mapping", split='CWE_1000')
cwe_map = {}
for d in cwe_1000:
    if d['CWE_ID'] not in cwe_map:
        cwe_map[d['CWE_ID']] = []
    cwe_map[d['CWE_ID']].extend(d['direct_parents']) 
    cwe_map[d['CWE_ID']].extend(d['direct_children'])
cwe_1000 = None
    
ground_truth_info = load_dataset("Leopo1d/OpenVul_Ground_Truth_Vulnerability_Information", split='ground_truth_info')
ground_truth = {}
for d in ground_truth_info:
    ground_truth[d['key']] = d
ground_truth_info = None

checklists = load_dataset("Leopo1d/OpenVul_Sample_Specification_for_RL_Reward_Evaluation", split='train')
specifications = {}
for d in checklists:
    specifications[d['key']] = {d['generated_specification'][0]['dimension']:d['generated_specification'][0]['description'],
                                d['generated_specification'][1]['dimension']:d['generated_specification'][1]['description'],
                                d['generated_specification'][2]['dimension']:d['generated_specification'][2]['description']}
checklists = None

system_prompt = "You are to act as a meticulous and impartial Code Security Expert and Evaluator."
evaluator = AsyncOpenAIAnalyzer(model="openai/gpt-oss-120b", system_prompt=system_prompt)

# CONCURRENCY_LIMIT = asyncio.Semaphore(1280)

async def detection_level_process_single_item(args):

    """Process a single prompt-completion pair"""
    completion, repetition, key = args
    
    is_vuln = True if "_vul" in key else False
    
    if repetition:
        vulnerability_reward = {
            "correctness": {
                "option": "INCORRECT",
                "justification": "repetition error"
            },
            "judge_reasoning": "repetition error"
        }

        return vulnerability_reward
    
    if '<think>' in completion and '</think>' in completion:
        completion = completion.split('</think>')[-1].strip()
        if (is_vuln and "HAS_VUL" in completion) or (not is_vuln and "NO_VUL" in completion):
            vulnerability_reward = {
                "correctness": {
                    "option": "CORRECT",
                    "justification": "match"
                },
                "judge_reasoning": "none"
            }
        else:
            vulnerability_reward = {
                "correctness": {
                    "option": "INCORRECT",
                    "justification": "mismatch"
                },
                "judge_reasoning": "none"
            }
            
    else:
        vulnerability_reward = {
            "correctness": {
                "option": "INCORRECT",
                "justification": "format error"
            },
            "judge_reasoning": "format error"
        }
    
    return vulnerability_reward

async def prediction_level_process_single_item(args):

    """Process a single prompt-completion pair"""
    completion, repetition, key, cwe_id = args
    
    is_vuln = True if "_vul" in key else False

    if cwe_id == '':
        new_cwe_id = []
    else:
        cwe_id = [ci.strip() for ci in cwe_id.split(",")]
        new_cwe_id = [ci for ci in cwe_id]
        for ci in cwe_id:
            if ci in cwe_map:
                new_cwe_id.extend(cwe_map[ci])
    
    if repetition:
        vulnerability_reward = {
            "correctness": {
                "option": "INCORRECT",
                "justification": "repetition error"
            },
            "judge_reasoning": "repetition error"
        }

        return vulnerability_reward
    
    if '<think>' in completion and '</think>' in completion:
        completion = completion.split('</think>')[-1].strip()
        matches = list(set(re.findall(r"CWE-\d+", completion)))
        if is_vuln:
            match = False
            for m in matches:
                if m in new_cwe_id:
                    match = True
                    break
            vulnerability_reward = {
                "correctness": {
                    "option": "CORRECT" if match else "INCORRECT",
                    "justification": "match" if match else "not match"
                },
                "judge_reasoning": "none"
            }
        else:
            match = True
            if "NO_VUL" in completion:
                match = True
            else:
                for m in matches:
                    if m in new_cwe_id:
                        match = False
                        break
            vulnerability_reward = {
                "correctness": {
                    "option": "CORRECT" if match else "INCORRECT",
                    "justification": "match" if match else "not match"
                },
                "judge_reasoning": "none"
            }
            
    else:
        vulnerability_reward = {
            "correctness": {
                "option": "INCORRECT",
                "justification": "format error"
            },
            "judge_reasoning": "format error"
        }
    
    return vulnerability_reward

async def reasoning_level_process_single_item(args):

    """Process a single prompt-completion pair"""
    completion, repetition, key, d = args
    
    if repetition:
        vulnerability_reward = {
            "correctness": {
                "option": "INCORRECT",
                "justification": "repetition error"
            },
            "reasoning_answer_consistency": {
                "option": "CONSISTENT",
                "justification": "repetition error"
            },
            "judge_reasoning": "repetition error"
        }
        return vulnerability_reward
    
    is_vuln = True if "_vul" in key else False
    if "_vul" in key:
        ground_truth_info = {"is_vulnerable": is_vuln,
                            "cve_description": d["cve_desc"],
                            "patch_commit_message": d["commit_msg"],
                            "patch_commit_diff": d['patch']}
    else:
        ground_truth_info = {"target_CVE_in_code": is_vuln,
                            "cve_description": d["cve_desc"],
                            "patch_commit_message": d["commit_msg"],
                            "patch_commit_diff": d['patch']}

    if '<think>' in completion and '</think>' in completion:
        completion = completion.split('</think>')[0].strip() + '\n' + '</think>' + '\n' + '<answer>' + '\n'+ completion.split('</think>')[-1].strip() + '\n' + '</answer>'
    else:
        last_has = completion.rfind("HAS_VUL")
        last_no = completion.rfind("NO_VUL")
        if last_has > last_no:
            answer = "HAS_VUL"
        elif last_no > last_has:
            answer = "NO_VUL"
        else:
            answer = ""
        if '<think>' not in completion and '</think>' in completion:
            completion = '<think>\n' + completion
            completion = completion.split('</think>')[0].strip() + '\n' + '</think>' + '\n' + '<answer>' + '\n'+ completion.split('</think>')[-1].strip() + '\n' + '</answer>'
        elif '<think>' in completion and '</think>' not in completion:
            completion = completion + '\n' + '</think>' + '\n' + '<answer>' + '\n' + answer + '\n' + '</answer>'
        else:
            completion = '<think>\n' + completion + '\n</think>' + '\n' + '<answer>' + '\n' + answer + '\n' + '</answer>'
                  
    input = json.dumps({"analysis": completion, "ground_truth_info": ground_truth_info}, ensure_ascii=False, indent=2)
    eval_prompt = construct_reasoning_level_judge_prompt(input, is_vuln=is_vuln)
    output = await evaluator.generate(eval_prompt)
    vulnerability_reward = await json_parser_reasoning(key, output, evaluator)
    
    return vulnerability_reward

async def specification_level_process_single_item(args):

    """Process a single prompt-completion pair"""
    completion, repetition, key, specification = args
    
    is_vuln = True if "_vul" in key else False

    if repetition:
        vulnerability_reward = {
            "correctness": {
                "option": "INCORRECT",
                "justification": "repetition error"
            },
            "localization": {
                "option": "INCORRECT",
                "justification": "repetition error"
            },
            "semantic": {
                "option": "INCORRECT",
                "justification": "repetition error"
            },
            "judge_reasoning": "repetition error"
        }

        return vulnerability_reward
    
    if '<think>' in completion and '</think>' in completion:
        completion = completion.split('</think>')[0].strip() + '\n' + '</think>' + '\n' + '<answer>' + '\n'+ completion.split('</think>')[-1].strip() + '\n' + '</answer>'
    else:
        last_has = completion.rfind("HAS_VUL")
        last_no = completion.rfind("NO_VUL")
        if last_has > last_no:
            answer = "HAS_VUL"
        elif last_no > last_has:
            answer = "NO_VUL"
        else:
            answer = ""
        if '<think>' not in completion and '</think>' in completion:
            completion = '<think>\n' + completion
            completion = completion.split('</think>')[0].strip() + '\n' + '</think>' + '\n' + '<answer>' + '\n'+ completion.split('</think>')[-1].strip() + '\n' + '</answer>'
        elif '<think>' in completion and '</think>' not in completion:
            completion = completion + '\n' + '</think>' + '\n' + '<answer>' + '\n' + answer + '\n' + '</answer>'
        else:
            completion = '<think>\n' + completion + '\n</think>' + '\n' + '<answer>' + '\n' + answer + '\n' + '</answer>'

    specification = json.dumps(specification, ensure_ascii=False, indent=2)
    eval_prompt = construct_specification_level_judge_prompt(is_vuln, specification, completion)
    output = await evaluator.generate(eval_prompt)
    vulnerability_reward = await json_parser_specification(key, output, evaluator)
    
    return vulnerability_reward

async def detection_level_main_reward(args_list):
    tasks = [detection_level_process_single_item(args) for args in args_list]
    return await asyncio.gather(*tasks)

async def prediction_level_main_reward(args_list):
    tasks = [prediction_level_process_single_item(args) for args in args_list]
    return await asyncio.gather(*tasks)

async def reasoning_level_async_main_reward(args_list):
    tasks = [reasoning_level_process_single_item(args) for args in args_list]
    return await asyncio.gather(*tasks)

async def specification_level_main_reward(args_list):
    tasks = [specification_level_process_single_item(args) for args in args_list]
    return await asyncio.gather(*tasks)

def detection_level_customized_reward(completions, completion_ids, **kwargs) -> list[float]:
    keys = kwargs['key']

    repetitions = [True if c[-1] != 151645 else False for c in completion_ids]

    args_list = [(completion, repetition, key) 
                    for completion, repetition, key in zip(completions, repetitions, keys)]
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            vulnerability_rewards = loop.run_until_complete(detection_level_main_reward(args_list))
        else:
            vulnerability_rewards = asyncio.run(detection_level_main_reward(args_list))
    except RuntimeError:
        # Fallback for creating new loop if get_event_loop fails
        vulnerability_rewards = asyncio.run(detection_level_main_reward(args_list))
    
    format_rewards = ["VALID" if '<think>' in content and '</think>' in content and ("HAS_VUL" in content.split('</think>')[-1] or "NO_VUL" in content.split('</think>')[-1]) and not ("HAS_VUL" in content.split('</think>')[-1] and "NO_VUL" in content.split('</think>')[-1]) else "INVALID" for content in completions]
       
    for i, r1 in enumerate(vulnerability_rewards):
        r1['format'] = {"option": format_rewards[i], "justification": None}
        
    correctness, format = [], []

    for i, r1 in enumerate(vulnerability_rewards):
        for k, v in r1.items():
            if k in ["correctness"]:
                if v['option'] == "CORRECT":
                    score = 1.0 
                else:
                    score = -1.0
                correctness.append(score)        

            elif k == "format":
                if v['option'] == "VALID":
                    score = 0.0
                else:
                    score = -0.2
                format.append(score)
                

    return correctness, format, vulnerability_rewards

def prediction_level_customized_reward(completions, completion_ids, **kwargs) -> list[float]:
    keys = kwargs['key']

    repetitions = [True if c[-1] != 151645 else False for c in completion_ids]

    args_list = [(completion, repetition, key, ground_truth['_'.join(key.split('_')[:-1])]['cwe_id']) 
                for completion, repetition, key in zip(completions, repetitions, keys)]
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            vulnerability_rewards = loop.run_until_complete(prediction_level_main_reward(args_list))
        else:
            vulnerability_rewards = asyncio.run(prediction_level_main_reward(args_list))
    except RuntimeError:
        # Fallback for creating new loop if get_event_loop fails
        vulnerability_rewards = asyncio.run(prediction_level_main_reward(args_list))

    format_rewards = ["VALID" if '<think>' in content and '</think>' in content and ("HAS_VUL" in content.split('</think>')[-1] or "NO_VUL" in content.split('</think>')[-1]) and not ("HAS_VUL" in content.split('</think>')[-1] and "NO_VUL" in content.split('</think>')[-1]) else "INVALID" for content in completions]
              
    for i, r1 in enumerate(vulnerability_rewards):
        r1['format'] = {"option": format_rewards[i], "justification": None}
        
    correctness, format = [], []

    for i, r1 in enumerate(vulnerability_rewards):
        for k, v in r1.items():
            if k == "correctness":
                if v['option'] == "CORRECT":
                    score = 1.0 
                else:
                    score = -1.0
                correctness.append(score)        
            elif k == "format":
                if v['option'] == "VALID":
                    score = 0.0
                else:
                    score = -0.2
                format.append(score)

    return correctness, format, vulnerability_rewards

def reasoning_level_customized_reward(completions, completion_ids, **kwargs) -> list[float]:
    keys = kwargs['key']

    repetitions = [True if c[-1] != 151645 else False for c in completion_ids]

    args_list = [(completion, repetition, key, ground_truth["_".join(key.split("_")[:-1])]) 
                 for completion, repetition, key in zip(completions, repetitions, keys)]
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            vulnerability_rewards = loop.run_until_complete(reasoning_level_async_main_reward(args_list))
        else:
            vulnerability_rewards = asyncio.run(reasoning_level_async_main_reward(args_list))
    except RuntimeError:
        # Fallback for creating new loop if get_event_loop fails
        vulnerability_rewards = asyncio.run(reasoning_level_async_main_reward(args_list))


    format_rewards = ["VALID" if '<think>' in content and '</think>' in content and ("HAS_VUL" in content.split('</think>')[-1] or "NO_VUL" in content.split('</think>')[-1]) and not ("HAS_VUL" in content.split('</think>')[-1] and "NO_VUL" in content.split('</think>')[-1]) else "INVALID" for content in completions]

    for i, r1 in enumerate(vulnerability_rewards):
        r1['format'] = {"option": format_rewards[i], "justification": None}
        
    correctness, format = [], []

    for i, r1 in enumerate(vulnerability_rewards):
        for k, v in r1.items():
            if k == "correctness":
                if v['option'] == "CORRECT":
                    score = 1.0 
                elif v['option'] == "UNKNOWN": # for non-vul
                    score = 1.0
                else:
                    score = -1.0
                correctness.append(score)        
            elif k == "format":
                if v['option'] == "VALID":
                    score = 0.0
                else:
                    score = -0.2
                format.append(score)
                
        for k, v in r1.items():
            if k == "reasoning_answer_consistency":
                if v['option'] == "INCONSISTENT":
                    correctness[-1] = -1.0
                    break

    return correctness, format, vulnerability_rewards

def specification_level_customized_reward(completions, completion_ids, **kwargs) -> list[float]:
    keys = kwargs['key']

    repetitions = [True if c[-1] != 151645 else False for c in completion_ids]

    args_list = [(completion, repetition, key, specifications[key]) 
                 for completion, repetition, key in zip(completions, repetitions, keys)]
    
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            vulnerability_rewards = loop.run_until_complete(specification_level_main_reward(args_list))
        else:
            vulnerability_rewards = asyncio.run(specification_level_main_reward(args_list))
    except RuntimeError:
        # Fallback for creating new loop if get_event_loop fails
        vulnerability_rewards = asyncio.run(specification_level_main_reward(args_list))

    format_rewards = ["VALID" if '<think>' in content and '</think>' in content and ("HAS_VUL" in content.split('</think>')[-1] or "NO_VUL" in content.split('</think>')[-1]) and not ("HAS_VUL" in content.split('</think>')[-1] and "NO_VUL" in content.split('</think>')[-1]) else "INVALID" for content in completions]
              
    for i, r1 in enumerate(vulnerability_rewards):
        r1['format'] = {"option": format_rewards[i], "justification": None}
        
    correctness, localization, semantic, format = [], [], [], []

    for i, r1 in enumerate(vulnerability_rewards):
        for k, v in r1.items():
            if k in ["correctness", "Verdict_Recall", "Verdict_Absence_of_Specific_Vuln"]:
                if v['option'] == "CORRECT":
                    score = 0.4 
                else:
                    score = -0.4
                correctness.append(score)        
            elif k in ["localization", "Evidence_Safeguard_Code", 'Evidence_Insecure_Code']:
                if v['option'] == "CORRECT":
                    score = 0.2
                elif v['option'] == "PARTIALLY CORRECT":
                    score = 0.1
                else:
                    score = 0.0
                localization.append(score)
            elif k in ["semantic", "Reasoning_Mechanism", "Reasoning_Resolution"]:
                if v['option'] == "CORRECT":
                    score = 0.4
                elif v['option'] == "PARTIALLY CORRECT":
                    score = 0.2
                else:
                    score = 0.0
                semantic.append(score)
            elif k == "format":
                if v['option'] == "VALID":
                    score = 0.0
                else:
                    score = -0.2
                format.append(score)

    return correctness, localization, semantic, format, vulnerability_rewards


def localization(completions: list[str], **kwargs) -> list[float]:
    pass

def semantic(completions: list[str], **kwargs) -> list[float]:
    pass

def format(completions: list[str], **kwargs) -> list[float]:
    pass


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_model_name_or_path (`str`, *optional*):
            Reward model id of a pretrained model hosted inside a model repo on huggingface.co or local path to a
            directory containing model weights saved using [`~transformers.PreTrainedModel.save_pretrained`].
        reward_funcs (`list[str]`, *optional*):
            Reward functions to use. Supported values are:

                - `"accuracy_reward"`
                - `"think_format_reward"`
                - `"get_soft_overlong_punishment"` (used value are `max_completion_len=1280`, `soft_punish_cache=256`)
                - any dotted import path " (e.g., `'my_lib.rewards.custom_reward'`).
    """

    reward_model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Reward model id of a pretrained model hosted inside a model repo on huggingface.co or "
            "local path to a directory containing model weights saved using `PreTrainedModel.save_pretrained`."
        },
    )
    reward_funcs: Optional[list[str]] = field(
        default=None,
        metadata={
            "help": "Reward functions to use. Supported values are: `accuracy_reward`, `think_format_reward`, "
            "`get_soft_overlong_punishment` (used value are `max_completion_len=1280`, `soft_punish_cache=256`), or "
            "any dotted import path (e.g., `'my_lib.rewards.custom_reward'`)."
        },
    )

from transformers import TrainerCallback

class SaveEpochEndCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        control.should_save = True
        return control
    
def main(script_args, training_args, model_args, other_args):
    # Get the reward models and functions
    training_args.name = other_args.name
    training_args.level = other_args.level
    training_args.judge_server_host = other_args.judge_server_host
    evaluator.client = AsyncOpenAI(base_url=f"http://{other_args.judge_server_host}:8002/v1",api_key="EMPTY")
    
    # Load the dataset
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        dtype=dtype
        # device_map={'': device_string},
    )
    
    REWARD_REGISTRY = {
        "detection": [detection_level_customized_reward, format],
        "prediction": [prediction_level_customized_reward, format],
        "reasoning": [reasoning_level_customized_reward, format],
        "specification": [specification_level_customized_reward, localization, semantic, format],
    }
    
    # Initialize the GRPO trainer
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=REWARD_REGISTRY[training_args.level],
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        #callbacks=[SaveEpochEndCallback()],
    )

    # Train the model
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint if training_args.resume_from_checkpoint else None)

    # Save and push to Hub
    trainer.save_model(training_args.output_dir)

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

def make_parser(subparsers: Optional[argparse._SubParsersAction] = None):
    dataclass_types = (GRPOScriptArguments, GRPOConfig, ModelConfig)
    if subparsers is not None:
        parser = subparsers.add_parser("grpo", help="Run the GRPO training script", dataclass_types=dataclass_types)
    else:
        parser = TrlParser(dataclass_types)

    parser.add_argument(
        "--name",
        type=str,
        default="default",
    )
    parser.add_argument(
        "--level",
        type=str,
        default="reasoning",
        choices=["detection", "prediction", "reasoning", "specification"],
    )
    parser.add_argument(
        "--judge_server_host",
        type=str,
    )
    return parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args, other_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args, other_args)
    dist.destroy_process_group()
