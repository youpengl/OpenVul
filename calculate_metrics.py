import json

def merge(fname):
    reward_path = f"results/{fname}-reward-gptoss120B.json"
    split_reward_path = f"results/{fname}-split-reward-gptoss120B.json"

    with open(split_reward_path, "r") as f:
        rewards = json.load(f)  

    new_rewards = {}
    for k, v in rewards.items():
        prefix_k = "_".join(k.split("_")[:-1])
        idx = int(k.split("_")[-1])
        if prefix_k not in new_rewards.keys():
            new_rewards[prefix_k] = [{}]*8
        new_rewards[prefix_k][idx] = v
            
    with open(reward_path, "w") as f:
        json.dump(new_rewards, f, ensure_ascii=False, indent=2)
        
def calculate_metrics(fname=None):

    with open(f"results/{fname}.json", 'r') as f:
        results = json.load(f)
    with open(f"results/{fname}-reward-gptoss120B.json", 'r') as f:
        judge = json.load(f)   
        
    TP, TN, FP, FN = 0, 0, 0, 0
    inconsistent = 0
    count, format_error = 0, 0
    diffs = {}
    correct_pairs = {}
    for k, v in results.items():
        pk = '_'.join(k.split('_')[:-1])
        if pk not in correct_pairs.keys():
            correct_pairs[pk] = [[], []]
        diffs[k] = 0
        for vi, vv in enumerate(v):
            count += 1
            if '<think>' not in vv or '</think>' not in vv or ('</think>' in vv and 'HAS_VUL' in vv.split('</think>')[-1] and 'NO_VUL' in vv.split('</think>')[-1]) or ('</think>' in vv and 'HAS_VUL' not in vv.split('</think>')[-1] and 'NO_VUL' not in vv.split('</think>')[-1]):
                format_error += 1
                if "_vul" in k:
                    FN += 1
                    correct_pairs[pk][0].append(0)
                else:
                    FP += 1
                    correct_pairs[pk][1].append(1)
            else:
                if judge[k][vi]['reasoning_answer_consistency']['option'] != "CONSISTENT":
                    inconsistent += 1
                if "_vul" in k:
                    if judge[k][vi]['correctness']['option'] == "CORRECT" and judge[k][vi]['reasoning_answer_consistency']['option'] == "CONSISTENT":
                        TP += 1
                        diffs[k] += 1          
                        correct_pairs[pk][0].append(1)
                    else:
                        FN += 1
                        correct_pairs[pk][0].append(0)
                else:
                    if judge[k][vi]['correctness']['option'] != "INCORRECT" and judge[k][vi]['reasoning_answer_consistency']['option'] == "CONSISTENT":
                        TN += 1
                        diffs[k] += 1
                        correct_pairs[pk][1].append(0)
                    else:
                        FP += 1
                        correct_pairs[pk][1].append(1)
                        
    passk = (TP+TN)/(TP+TN+FN+FP)
    recall = TP/(TP+FN)
    precision = TP/(TP+FP)
    F1 = (2*recall*precision)/(recall+precision)
    FPR = FP/(FP+TN)
    FNR = FN/(FN+TP)
    TNR = TN/(TN+FP)
    fc = 1-format_error/count
    pass8 = sum([1 for k, v in diffs.items() if v != 0])/len(diffs)
    all_correct = sum([1 for k, v in diffs.items() if v == 8])/len(diffs)

    print(f"{fname}: #test {count}")        
    print(f"Pass@1: {passk*100:.2f}, Pass@8: {pass8*100:.2f}, Recall: {recall*100:.2f}, Precision: {precision*100:.2f}, F1: {F1*100:.2f}, Format: {(1-fc)*100:.2f}|{fc*100:.2f}, Consistency: {inconsistent/(TP+TN+FN+FP)*100:.2f}|{(1-inconsistent/(TP+TN+FN+FP))*100:.2f}, FPR: {FPR*100:.2f}, FNR: {FNR*100:.2f}, TNR: {TNR*100:.2f}")
    
    dif = []
    all_correct, all_p, all_n, all_wrong = 0, 0, 0, 0
    for k, v in correct_pairs.items():
        vul = v[0]
        if sum(vul) == 0:
            dif.append(k)
        patched = v[1]
        all_correct += sum([1 for a, b in zip(vul, patched) if a == 1 and b == 0])
        all_p += sum([1 for a, b in zip(vul, patched) if a == 1 and b == 1])
        all_n += sum([1 for a, b in zip(vul, patched) if a == 0 and b == 0])
        all_wrong += sum([1 for a, b in zip(vul, patched) if a == 0 and b == 1])

    all = all_correct+all_p+all_n+all_wrong
    
    print(f"P-C: {all_correct/all*100:.2f}, P-B: {all_n/all*100:.2f}, P-V: {all_p/all*100:.2f}, P-R: {all_wrong/all*100:.2f}")
    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}, inconsistent: {inconsistent}, format_error: {format_error}")
    print(f"all_correct: {all_correct}, all_p: {all_p}, all_n: {all_n}, all_wrong: {all_wrong}, all: {all}")

merge("INPUT YOUR FILE NAME")
calculate_metrics("INPUT YOUR FILE NAME")
