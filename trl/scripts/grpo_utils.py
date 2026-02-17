from openai import AsyncOpenAI
import json
import ast
import asyncio

def construct_reasoning_level_judge_prompt(
    input: dict,
    is_vuln: bool,
):

    if is_vuln:

        prompt = \
"""1. GOAL
Your primary goal is to assess the quality of an analysis of a vulnerable piece of code. You must evaluate the analysis against a provided set of ground truth information. Your judgment must be objective, strictly adhering to the provided option rubric and based only on the information given.

2. INPUT FORMAT
You will be provided with a JSON object containing two main keys: analysis and ground_truth_info

```json
{
"analysis": "<The full analysis, including its reasoning and answer.>",
"ground_truth_info": {
    "is_vulnerable": true,
    "cve_description": "<The official CVE description of the vulnerability.>",
    "patch_commit_message": "<The developer's commit message that may explain the vulnerability.>"
    "patch_commit_diff": "<A git-style diff showing the changes from the pre-patched (vulnerable) to the post-patched (non-vulnerable) code.>"
}
}
```

Analysis Context & Critical Warning:
The analysis you are evaluating is generated based on the pre-patched (vulnerable) code. The patch_commit_diff and patch_commit_message is provided only as a reference to help you understand the precise location and nature of the ground truth vulnerability. Do not let it mislead you into thinking the vulnerability has already been fixed in the pre-patched code that is analyzed.

3. EVALUATION WORKFLOW AND OPTION RUBRIC

You must follow these steps to evaluate the analysis and produce a final JSON output. For each dimension, you need to provide a brief justification and choose an option.

Step 1: Analyze Ground Truth
First, carefully review all the information in the ground_truth_info. This is your foundation for judgment.

Step 2: Evaluate Each Dimension
Assess the analysis across the following dimensions. Choose an option for each based on the rubric below.

Dimension 1: Correctness
Task: Evaluate if the analysis correctly identifies the target CVE mentioned in the ground_truth_info.

Option Rubric:
* CORRECT: The analysis identifies the code as vulnerable, AND the explanation of the root cause of the predicted vulnerability also aligns with the ground truth vulnerability information provided in the ground_truth_info. Besides, it is acceptable if the analysis also identifies other vulnerabilities.
* PARTIALLY CORRECT: The analysis identifies the code as vulnerable, BUT the explanation of the root cause of the predicted vulnerability does not align with the ground truth vulnerability information provided in the ground_truth_info.
* INCORRECT: The analysis identifies the code as non-vulnerable.

Dimension 2: Reasoning-Answer Consistency
Task: Compare whether the reasoning (within the <think> tags) contradict the answer (within the <answer> tags). Please evaluate solely based on the consistency between the reasoning and the answer. DO NOT consider the correctness of the prediction (i.e., whether it matches the ground truth) for this dimension.

Option Rubric:
* CONSISTENT: The reasoning within the <think> tags and the answer within the <answer> tags are consistent; both indicate it has a vulnerability (HAS_VUL) or it does not have a vulnerability (NO_VUL).
* INCONSISTENT: The reasoning within the <think> tags and the answer within the <answer> tags are not consistent; one indicates it has a vulnerability (HAS_VUL) while the other indicates it does not have a vulnerability (NO_VUL).

4. OUTPUT FORMAT

Your final output must be a single JSON object. Do not include any text or explanation outside of the JSON structure. The JSON must contain a key for each dimension's justification and option.

```json
{
    "correctness": {
        "justification": "<Your brief reason>",
        "option": <choose from ["CORRECT", "PARTIALLY CORRECT", "INCORRECT"]>
    },
    "reasoning_answer_consistency": {
        "justification": "<Your brief reason>",
        "option": <choose from ["CONSISTENT", "INCONSISTENT"]>
    }
}
```"""+f"""

CURRENT INPUT

```json
{input}
```
"""
    
    else:
        prompt = \
"""1. GOAL
Your primary goal is to assess the quality of an analysis of post-patched code in which the target CVE has been fixed. You must evaluate the analysis against a provided set of ground truth information. Your judgment must be objective, strictly adhering to the provided option rubric and based only on the information given.

2. INPUT FORMAT
You will be provided with a JSON object containing two main keys: analysis and ground_truth_info

```json
{
"analysis": "<The full analysis, including its reasoning and answer.>",
"ground_truth_info": {
    "target_CVE_in_code": false,
    "cve_description": "<The official CVE description of the vulnerability that was fixed.>",
    "patch_commit_message": "<The developer's commit message that may explain the fix.>",
    "patch_commit_diff": "<A git-style diff showing the changes that fixed the vulnerability.>"
}
}
```

Analysis Context & Critical Warning:
The analysis you are evaluating is generated based on the post-patched code in which the target CVE has been fixed. The ground_truth_info is provided only as a reference to help you understand how the target CVE is fixed. Do not let it mislead you into thinking the target CVE is still present in the post-patched code that is analyzed. Please note that "target_CVE_in_code": false in the ground_truth_info can only indicate that the target CVE does not exist in the code, but it cannot guarantee whether the code contains other unknown vulnerabilities.

3. EVALUATION WORKFLOW AND OPTION RUBRIC

You must follow these steps to evaluate the analysis and produce a final JSON output. For each dimension, you need to provide a brief justification and choose an option.

Step 1: Analyze Ground Truth
First, carefully review all the information in the ground_truth_info. This is your foundation for judgment.

Step 2: Evaluate Each Dimension
Assess the analysis across the following dimensions. Choose an option for each based on the rubric below.

Dimension 1: Correctness
Task: Evaluate whether the analysis identifies that a vulnerability with the exactly same root cause in ground_truth_info still exists in the post-patched code.

Option Rubric:
* CORRECT: The analysis finds no vulnerabilities in the code.
* UNKNOWN: The analysis does not identify a vulnerability with the exactly same root cause as in ground_truth_info, but it does identify other unknown vulnerabilities with different root causes.
* INCORRECT: Select this option ONLY IF the analysis identifies that a vulnerability with the exactly same root cause as in ground_truth_info still exists in the code. Please select UNKNOWN if the analysis identifies vulnerabilities whose root causes are not exactly the same as the vulnerability in ground_truth_info, even if they are only similar. For example, the analysis identifies that the code contains an out-of-bound access vulnerability, and the target CVE in ground_truth_info is also an out-of-bound access vulnerability. However, the root causes of the two vulnerabilities are not exactly same (e.g., they occur in different locations). In this situation, you should choose UNKNOWN, because the two vulnerabilities are not exactly the same.

Dimension 2: Reasoning-Answer Consistency
Task: Compare whether the reasoning (within the <think> tags) contradict the answer (within the <answer> tags). Please evaluate solely based on the consistency between the reasoning and the answer. DO NOT consider the correctness of the prediction (i.e., whether it matches the ground truth) for this dimension.

Option Rubric:
* CONSISTENT: The reasoning within the <think> tags and the answer within the <answer> tags are consistent; both indicate it has a vulnerability (HAS_VUL) or it does not have a vulnerability (NO_VUL).
* INCONSISTENT: The reasoning within the <think> tags and the answer within the <answer> tags are not consistent; one indicates it has a vulnerability (HAS_VUL) while the other indicates it does not have a vulnerability (NO_VUL).

4. OUTPUT FORMAT

Your final output must be a single JSON object. Do not include any text or explanation outside of the JSON structure. The JSON must contain a key for each dimension's justification and option.

```json
{
    "correctness": {
        "justification": "<Your brief reason>",
        "option": <choose from ["CORRECT", "UNKNOWN", "INCORRECT"]>
    },
    "reasoning_answer_consistency": {
        "justification": "<Your brief reason>",
        "option": <choose from ["CONSISTENT", "INCONSISTENT"]>
    }
}
```"""+f"""

CURRENT INPUT

```json
{input}
```
"""
        
    return prompt

def construct_specification_level_judge_prompt(
    is_vuln,
    checklist,
    analysis
):

    if is_vuln:

        prompt = \
f"""
1. GOAL:
Your task is to evaluate a vulnerability analysis for a code sample known to be HAS_VUL (vulnerable). Verify if the detector successfully identified the specific target vulnerability described in the Checklist.

2. SCORING RUBRIC:

Dimension 1: Verdict (Strict Binary)
* CORRECT: The analysis identifies the specific vulnerability type requested in the checklist (e.g., "Buffer Overflow"). Ignore any extra/unrelated vulnerabilities listed by the detector. As long as the target is found, it is true.
* INCORRECT: The analysis fails to mention the target vulnerability type.

Dimension 2: Evidence (Insecure Code Snippet)
* CORRECT: The analysis quotes the exact code snippet (or a semantically identical variation) requested in the checklist.
* PARTIALLY CORRECT: The analysis cites the correct variable name or surrounding logic, but does not quote the specific snippet exactly. Or, it quotes a huge block of code that contains the snippet but lacks precision.
* INCORRECT: No specific evidence or incorrect code cited.

Dimension 3: Reasoning (Attack Mechanism)
* CORRECT: The analysis correctly explains both the root cause (e.g., "missing check") and the consequence (e.g., "overflow").
* PARTIALLY CORRECT: Explains the general issue (e.g., "unsafe copy") but misses technical details.
* INCORRECT: Incorrect or generic reasoning.

3. INPUT:

The Ground Truth Checklist (Target: Vulnerable):

{checklist}

The Candidate Analysis:

{analysis}

4. OUTPUT FORMAT (JSON ONLY):"""+"""

```json
{
    "Verdict_Recall": {
        "justification": "<Your brief reason>",
        "option": <choose from ["CORRECT", "INCORRECT"]>
    },
    "Evidence_Insecure_Code": {
        "justification": "<Your brief reason>",
        "option": <choose from ["CORRECT", "PARTIALLY CORRECT", "INCORRECT"]>
    },
    "Reasoning_Mechanism": {
        "justification": "<Your brief reason>",
        "option": <choose from ["CORRECT", "PARTIALLY CORRECT", "INCORRECT"]>
    }
}
```"""

    else:
        prompt = \
f"""
1. GOAL:
Your task is to evaluate a vulnerability analysis for a code sample known to be NO_VUL (vulnerability fixed). Verify if the detector correctly concludes the code is safe regarding the specific vulnerability.

2. SCORING RUBRIC:

Dimension 1: Verdict (Strict Binary)
* CORRECT: The analysis says "NO_VUL" or The analysis reports other vulnerabilities but does not list the specific fixed vulnerability mentioned in the checklist.
* INCORRECT: The analysis explicitly claims the specific target vulnerability (e.g., the one described in the checklist) still exists.

Dimension 2: Evidence (Safeguard Snippet)
* CORRECT: The analysis quotes the exact safeguard code (e.g., the new check/sanitizer) requested in the checklist.
* PARTIALLY CORRECT: The analysis cites the correct variable name or surrounding logic, but does not quote the specific snippet exactly. Or, it quotes a huge block of code that contains the snippet but lacks precision.
* INCORRECT: No specific evidence or incorrect code cited.

Dimension 3: Reasoning (Safety Logic)
* CORRECT: Explains why the code is safe (e.g., "The new check prevents the overflow").
* PARTIALLY CORRECT: Vague acknowledgment of safety without specific logic.
* INCORRECT: Incorrect logic or claims the code is unsafe.

3. INPUT:
The Ground Truth Checklist (Target: Safe/Fixed):

{checklist}

The Candidate Analysis:

{analysis}

4. OUTPUT FORMAT (JSON ONLY):"""+"""

```json
{
    "Verdict_Absence_of_Specific_Vuln": {
        "justification": "<Your brief reason>",
        "option": <choose from ["CORRECT", "INCORRECT"]>
    },
    "Evidence_Safeguard_Code": {
        "justification": "<Your brief reason>",
        "option": <choose from ["CORRECT", "PARTIALLY CORRECT", "INCORRECT"]>
    },
    "Reasoning_Resolution": {
        "justification": "<Your brief reason>",
        "option": <choose from ["CORRECT", "PARTIALLY CORRECT", "INCORRECT"]>
    }
}
```"""
        
    return prompt

CHECK_PROMPT_reasoning = """You are a JSON format calibrator. Please extract the data from INPUT following the output template below. Please exclude the key "justification" and its value.

OUTPUT TEMPLATE
```json
{
    "correctness": {
        "option": [extracted content]
    },
    "reasoning_answer_consistency": {
        "option": [extracted content]
    }
}
```"""

CHECK_PROMPT_specification_vul = """You are a JSON format calibrator. Please extract the data from INPUT following the output template below. Please exclude the key "justification" and its value.

OUTPUT TEMPLATE
```json
{
    "Verdict_Recall": {
        "option": [extracted content],
    },
    "Evidence_Insecure_Code": {
        "option": [extracted content],
    },
    "Reasoning_Mechanism": {
        "option": [extracted content],
    },
```"""

CHECK_PROMPT_specification_patched = """You are a JSON format calibrator. Please extract the data from INPUT following the output template below. Please exclude the key "justification" and its value.

OUTPUT TEMPLATE
```json
{
    "Verdict_Absence_of_Specific_Vuln": {
        "option": [extracted content],
    },
    "Evidence_Safeguard_Code": {
        "option": [extracted content],
    },
    "Reasoning_Resolution": {
        "option": [extracted content],
    }
}
```"""

def str2json_reasoning(key, output):
    if output == 'Error':
        print(f"{key}, !!!!!!!! 3 try fails")       
        vulnerability_reward = {
            "correctness": {
                "option": "INCORRECT",
                "justification": "response error"
            },
            "reasoning_answer_consistency": {
                "option": "CONSISTENT",
                "justification": "response error"
            },
            "judge_reasoning": "response error"
        }
    else:
        reasoning = output['reasoning']
        output = output['answer']
        try:
            vulnerability_reward = json.loads(output.strip())
        except:
            try:
                vulnerability_reward = ast.literal_eval(output.split('```json')[1].split("```")[0].strip())
            except:
                try:
                    vulnerability_reward = output.split('```json')[1].split("```")[0].strip()
                    vulnerability_reward = f"r'''{vulnerability_reward}'''"
                    vulnerability_reward = vulnerability_reward.split("r'''")[-1].split("'''")[0]
                    vulnerability_reward = json.loads(vulnerability_reward)
                except:
                    try:
                        vulnerability_reward = json.loads(output.split('```json')[1].split("```")[0].strip())
                    except:
                        print(f"{key}, !!!!!!!! cannot extract from:\n{output}")
                        vulnerability_reward = {}
                        
        if vulnerability_reward != {}:
            vulnerability_reward["judge_reasoning"] = reasoning
                    
    return vulnerability_reward

def str2json_specification(key, output):
    if output == 'Error':
        print(f"{key}, !!!!!!!! 3 try fails")
        
        vulnerability_reward = {
            "correctness": {
                "option": "INCORRECT",
                "justification": "response error"
            },
            "localization": {
                "option": "INCORRECT",
                "justification": "response error"
            },
            "semantic": {
                "option": "INCORRECT",
                "justification": "response error"
            },
            "judge_reasoning": "response error"
        }
        return vulnerability_reward
    else:
        reasoning = output['reasoning']
        output = output['answer']
        try:
            vulnerability_reward = json.loads(output.strip())
        except:
            try:
                vulnerability_reward = ast.literal_eval(output.split('```json')[1].split("```")[0].strip())
            except:
                try:
                    vulnerability_reward = output.split('```json')[1].split("```")[0].strip()
                    vulnerability_reward = f"r'''{vulnerability_reward}'''"
                    vulnerability_reward = vulnerability_reward.split("r'''")[-1].split("'''")[0]
                    vulnerability_reward = json.loads(vulnerability_reward)
                except:
                    try:
                        vulnerability_reward = json.loads(output.split('```json')[1].split("```")[0].strip())
                    except:
                        print(f"{key}, !!!!!!!! cannot extract from:\n{output}")
                        vulnerability_reward = {}
                        
        if vulnerability_reward != {}:
            vulnerability_reward["judge_reasoning"] = reasoning

        return vulnerability_reward

class AsyncOpenAIAnalyzer():
    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        system_prompt="",
        temperature=0,
        client=None,
    ):
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature

    async def generate(
        self,
        prompt,
        system_prompt=None
    ) -> str:

        messages = [
            {"role": "developer", "content": system_prompt if system_prompt != None else self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        max_retries = 3
        timeout = 600
        for attempt in range(max_retries):
            try:
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        extra_body={"reasoning_effort": "high"}
                    ),
                    timeout=timeout+300*attempt
                )

                return {"reasoning": response.choices[0].message.model_extra['reasoning_content'], "answer": response.choices[0].message.content}

            except asyncio.TimeoutError:
                print(f"[Timeout] {attempt+1}/{max_retries}")

            except Exception as e:
                print(f"[API Error: {e}] retry {attempt+1}/{max_retries}")

            # use sub_clients
            sub_clients = {}
            with open("candidate_judge_server.txt", "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        sub_clients[line.strip()] = AsyncOpenAI(base_url=f"http://{line.strip()}:8002/v1",api_key="EMPTY")
                        print(f"sub clients {line.strip()} init successful!")
                    except:
                        print(f"sub clients {line.strip()} init error!")
                        
            if sub_clients != {}:
                for cname, c in sub_clients.items():
                    print(f"[sub_client: {cname}] activated!")
                    try:
                        response = await asyncio.wait_for(
                            c.chat.completions.create(
                                model=self.model,
                                messages=messages,
                                temperature=self.temperature,
                                extra_body={"reasoning_effort": "high"}
                            ),
                            timeout=timeout+300*attempt
                        )

                        return {"reasoning": response.choices[0].message.model_extra['reasoning_content'], "answer": response.choices[0].message.content}

                    except asyncio.TimeoutError:
                        print(f"[Timeout] {attempt+1}/{max_retries}")

                    except Exception as e:
                        print(f"[API Error: {e}] retry {attempt+1}/{max_retries}")
            
        return "Error"

async def json_parser_specification(key, output, evaluator):
    vulnerability_reward = str2json_specification(key, output)
    if vulnerability_reward == {}:
        # await generate
        print(f"{key}, !!!!!!!! parse error, try to use LLM!")
        if '_vul' in key:
            new_output = await evaluator.generate(f"INPUT\n{output['answer']}", system_prompt=CHECK_PROMPT_specification_vul)
        else:
            new_output = await evaluator.generate(f"INPUT\n{output['answer']}", system_prompt=CHECK_PROMPT_specification_patched)
        vulnerability_reward = str2json_specification(key, new_output)
        if vulnerability_reward == {}:
            print(f"{key}, !!!!!!!! still fail to parse the json using LLM")

            vulnerability_reward = {
                "correctness": {
                    "option": "INCORRECT",
                    "justification": "parse error"
                },
                "localization": {
                    "option": "INCORRECT",
                    "justification": "parse error"
                },
                "semantic": {
                    "option": "INCORRECT",
                    "justification": "parse error"
                },
                "judge_reasoning": "parse error"
            }                
    return vulnerability_reward

async def json_parser_reasoning(key, output, evaluator):
    vulnerability_reward = str2json_reasoning(key, output)
    if vulnerability_reward == {}:
        # await generate
        print(f"{key}, !!!!!!!! parse error, try to use LLM!")
        new_output = await evaluator.generate(f"INPUT\n{output['answer']}", system_prompt=CHECK_PROMPT_reasoning)
        vulnerability_reward = str2json_reasoning(key, new_output)
        if vulnerability_reward == {}:
            print(f"{key}, !!!!!!!! still fail to parse the json using LLM")
            vulnerability_reward = {
                "correctness": {
                    "option": "INCORRECT",
                    "justification": "parse error"
                },
                "reasoning_answer_consistency": {
                    "option": "CONSISTENT",
                    "justification": "parse error"
                },
                "judge_reasoning": "parse error"
            }
                        
    return vulnerability_reward