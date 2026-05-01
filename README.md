<div align="center">
  <h1>OpenVul: An Open-Source Post-Training Framework for LLM-Based Vulnerability Detection</h1>
    <p>
    <img src="https://img.shields.io/badge/python-3.11.13-blue" alt="Python">
    <img src="https://img.shields.io/badge/license-Apache 2.0-green" alt="License">
    <a href="https://huggingface.co/collections/Leopo1d/openvul"><img src="https://img.shields.io/badge/datasets & models-Hugging Face-yellow?style=flat&logo=huggingface" alt="asc"></a>
  </p>
  
</div>

## 💥 News
- The corresponding paper, “From SFT to RL: Demystifying the Post-Training Pipeline for LLM-Based Vulnerability Detection,” is available on [arXiv](https://arxiv.org/abs/2602.14012).


## 🛠️ Environment Setup
```bash
git clone https://github.com/youpengl/OpenVul.git
cd OpenVul
pip install uv
uv python install 3.11.13
uv venv --python 3.11.13
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install flash-attn==2.8.1 --no-build-isolation
export HF_TOKEN = ""
export WANDB_API_KEY = ""
```

## ⚙️ Post-training Framework

We have developed the first post-training framework for LLM-based VD based on the [Hugging Face TRL](https://github.com/huggingface/trl) library. Our framework currently supports SFT, Preference Optimization (e.g., DPO, ORPO), and on-policy RL (e.g., GRPO) for VD LLMs. We plan to continuously integrate more specialized post-training algorithms for VD in the future. 


### 🏃🏻 Running Details
```bash
# Train

## Cold Start Stage
sbatch sft.slurm


## Preference Optimizaiton
sbatch dpo.slurm
sbatch orpo.slurm


## RL Stage

### Step 1: Run judge server and vllm server
sbatch judge_server.slurm
sbatch vllm_server.slurm

### Step 2: GRPO Training
#### Please switch between ['detection', 'prediction', 'reasoning', 'specification'] to change the reward system in the file `grpo.sh`.
#### For specification-based reward, please modify: --reward_weights 1.0 1.0 1.0 1.0.
#### By defualt, the reasoning-based reward is recommended to use to balance model performance and training stability.
sbatch grpo.slurm


# Test

## LLM Inference via vLLM
sbatch vllm_inference.slurm


## Output Judge
python LLM_judge_for_vulnerability_detection.py --gpu [input your gpu node ip] --name [input your model name] 


## Metric Calculation
python calculate_metrics.py

```

### 💻 GPU Requirements



| Stage | Purpose | Hardware (A100 80GB) | Estimated Duration |
| :--- | :--- | :--- | :--- |
| **Cold Start** | SFT | 4x GPUs | < 1 Days |
| **Preference Optimization** | DPO / ORPO | 4x GPUs | < 1 Days |
| **RL Stage (Training)** | GRPO Training | 8x GPUs | 3 - 5 Days |
| **RL Stage (Judge Server)** | Reward Model / LLM-as-a-Judge | 4x GPUs | Synchronous |
| **RL Stage (vLLM Server)** | Rollout / Inference | 2x GPUs | Synchronous |

## 🗂️ Overview of the Datasets Released on [Hugging Face](https://huggingface.co/collections/Leopo1d/openvul)
- **[OpenVul](https://huggingface.co/datasets/Leopo1d/OpenVul):** The originally collected dataset.

- **[OpenVul_Distilled_Vulnerability_Reasoning_CoTs_from_DeepSeek-R1-0528](https://huggingface.co/datasets/Leopo1d/OpenVul_Distilled_Vulnerability_Reasoning_CoTs_from_DeepSeek-R1-0528):** This dataset provides all training data's vulnerability reasoning CoTs (with 8 generations per sample) distilled from DeepSeek-R1-0528. This dataset has not been filtered for correctness and can be used to construct vulnerability reasoning and preference datasets for future research.

- **[OpenVul_Rejection_Sampling_based_Vulnerability_Reasoning_Dataset_for_SFT](https://huggingface.co/datasets/Leopo1d/OpenVul_Rejection_Sampling_based_Vulnerability_Reasoning_Dataset_for_SFT):** This dataset provides high-quality, correctness-filtered vulnerability reasoning data to support the SFT of specialized VD LLMs for future research.

- **[OpenVul_Rationalization_based_Vulnerability_Reasoning_Dataset_for_SFT](https://huggingface.co/datasets/Leopo1d/OpenVul_Rationalization_based_Vulnerability_Reasoning_Dataset_for_SFT):** This dataset provides all training data's vulnerability reasoning CoTs  (with one generation per sample) distilled from DeepSeek-R1-0528, collected using a rationalization-based data curation method.

- **[OpenVul_Vulnerability_Preference_Dataset_for_ORPO](https://huggingface.co/datasets/Leopo1d/OpenVul_Vulnerability_Preference_Dataset_for_ORPO):** This dataset provides high-quality vulnerability preference data, selected from the **[OpenVul_Distilled_Vulnerability_Reasoning_CoTs_from_DeepSeek-R1-0528](https://huggingface.co/datasets/Leopo1d/OpenVul_Distilled_Vulnerability_Reasoning_CoTs_from_DeepSeek-R1-0528)**, to support the preference optimization (e.g., ORPO) of specialized VD LLMs in future research.

- **[OpenVul_Vulnerability_Preference_Dataset_for_DPO](https://huggingface.co/datasets/Leopo1d/OpenVul_Vulnerability_Preference_Dataset_for_DPO):** This dataset provides high-quality vulnerability preference data, curated from vulnerability reasoning CoTs distilled from the SFT LLM **[OpenVul-Qwen3-4B-SFT-ep5](https://huggingface.co/Leopo1d/OpenVul-Qwen3-4B-SFT-ep5)**, to support the preference optimization (e.g., DPO) of specialized VD LLMs for future research.

- **[OpenVul_Vulnerability_Query_Dataset_for_RL](https://huggingface.co/datasets/Leopo1d/OpenVul_Vulnerability_Query_Dataset_for_RL):** This dataset provides context-aware vulnerability queries partitioned chronologically by commit date into training, validation, and test sets, designed to support the RL (e.g., GRPO) of specialized VD LLMs in future research.

- **[OpenVul_Ground_Truth_Vulnerability_Information](https://huggingface.co/datasets/Leopo1d/OpenVul_Ground_Truth_Vulnerability_Information):** This dataset provides ground truth vulnerability information (CWE ID, CVE description, commit message, and patch diff) for all samples in the OpenVul_Vulnerability_Query_Dataset_for_RL collection, enabling multi-granular model reward evaluation and performance evaluation.

- **[OpenVul_CWE_Hierarchical_Mapping](https://huggingface.co/datasets/Leopo1d/OpenVul_CWE_Hierarchical_Mapping)** This dataset provides the direct hierarchical (parent-child) relationships for all CWEs in the CWE-1000 Research view, designed to support prediction-level CWE matching.

- **[OpenVul_Sample_Specification_for_RL_Reward_Evaluation](https://huggingface.co/datasets/Leopo1d/OpenVul_Sample_Specification_for_RL_Reward_Evaluation)** This dataset provides generated specifications for each training sample to facilitate specification-based reward evaluation and sample-level judgment, moving beyond traditional coarse-grained ground truth labels like binary indicators or CWE IDs.

## 🧠 Overview of the Models Released on [Hugging Face](https://huggingface.co/collections/Leopo1d/openvul)

- **OpenVul-Qwen3-4B-SFT**, fine-tuned from Qwen3-4B on the **[OpenVul_Rejection_Sampling_based_Vulnerability_Reasoning_Dataset_for_SFT](https://huggingface.co/datasets/Leopo1d/OpenVul_Rejection_Sampling_based_Vulnerability_Reasoning_Dataset_for_SFT)**, serves as the foundational backbone for VD. It has been fine-tuned on high-quality vulnerability reasoning CoTs to establish basic security expertise and instruction-following capabilities. Three checkpoints, **[OpenVul-Qwen3-4B-SFT-ep1](https://huggingface.co/Leopo1d/OpenVul-Qwen3-4B-SFT-ep1)**, **[OpenVul-Qwen3-4B-SFT-ep3](https://huggingface.co/Leopo1d/OpenVul-Qwen3-4B-SFT-ep3)**, **[OpenVul-Qwen3-4B-SFT-ep5](https://huggingface.co/Leopo1d/OpenVul-Qwen3-4B-SFT-ep5)**, are available.

- **[OpenVul-Qwen3-4B-DPO](https://huggingface.co/Leopo1d/OpenVul-Qwen3-4B-DPO)**, post-trained from **[OpenVul-Qwen3-4B-SFT-ep5](https://huggingface.co/Leopo1d/OpenVul-Qwen3-4B-SFT-ep5)** on the **[OpenVul_Vulnerability_Preference_Dataset_for_DPO](https://huggingface.co/datasets/Leopo1d/OpenVul_Vulnerability_Preference_Dataset_for_DPO)**, serves as an advanced VD LLM optimized to distinguish between vulnerable code and its patched counterparts without an explicit reward model.

- **[OpenVul-Qwen3-4B-ORPO](https://huggingface.co/Leopo1d/OpenVul-Qwen3-4B-ORPO)** post-trained from **[OpenVul-Qwen3-4B-SFT-ep5](https://huggingface.co/Leopo1d/OpenVul-Qwen3-4B-SFT-ep5)** on the **[OpenVul_Vulnerability_Preference_Dataset_for_ORPO](https://huggingface.co/datasets/Leopo1d/OpenVul_Vulnerability_Preference_Dataset_for_ORPO)**, serves as an advanced VD LLM optimized to distinguish between vulnerable code and its patched counterparts without reference and reward models.

- **[OpenVul-Qwen3-4B-GRPO](https://huggingface.co/Leopo1d/OpenVul-Qwen3-4B-GRPO)**, post-trained from **[OpenVul-Qwen3-4B-SFT-ep3](https://huggingface.co/Leopo1d/OpenVul-Qwen3-4B-SFT-ep3)** on the **[OpenVul_Vulnerability_Query_Dataset_for_RL](https://huggingface.co/datasets/Leopo1d/OpenVul_Vulnerability_Query_Dataset_for_RL)**, serves as the state-of-the-art (SOTA) specialized VD reasoning LLM, utilizing on-policy RL to navigate complex vulnerability reasoning paths.

## 📚 Citation
```
@misc{li2026sftrldemystifyingposttraining,
      title={From SFT to RL: Demystifying the Post-Training Pipeline for LLM-based Vulnerability Detection}, 
      author={Youpeng Li and Fuxun Yu and Xinda Wang},
      year={2026},
      eprint={2602.14012},
      archivePrefix={arXiv},
      primaryClass={cs.CR},
      url={https://arxiv.org/abs/2602.14012},
}
```

## 📬 Contact
Feel free to contact me via youpeng [dot] li [dot] utdallas [dot] edu
