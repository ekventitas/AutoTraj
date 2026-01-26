import json
import re

# --------- 答案判定函数 ---------
def is_answer_correct(gold_answer, traj_answer):
    def extract_brace_content(text):
        matches = re.findall(r'\{(.*?)\}', text)
        if matches:
            return matches[-1].strip()
        return text.strip()

    gold_ans = extract_brace_content(gold_answer)
    traj_ans = extract_brace_content(traj_answer)

    try:
        gold_num = float(re.sub(r'[^\d.-]', '', gold_ans))
        traj_num = float(re.sub(r'[^\d.-]', '', traj_ans))
        return abs(gold_num - traj_num) < 1e-6
    except:
        if gold_ans == traj_ans:
            return True
        if traj_ans in gold_ans:
            return True
    return False


# --------- 主处理函数（返回统计信息）---------
def classify_trajectories(input_file, correct_file, incorrect_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    correct_data = []
    incorrect_data = []

    correct_traj_count = 0
    incorrect_traj_count = 0

    for item in data:
        question = item['question']
        gold_answer = item['gold_answer']
        trajectories = item.get('trajectories', [])

        traj_classes = []
        for traj in trajectories:
            match = re.search(r'<answer>(.*?)</answer>', traj['react_trajectory'], re.DOTALL)
            traj_ans_text = match.group(1) if match else ''
            traj_classes.append(is_answer_correct(gold_answer, traj_ans_text))

        correct_traj_count += sum(traj_classes)
        incorrect_traj_count += len(traj_classes) - sum(traj_classes)

        if all(traj_classes):
            correct_data.append(item)
        elif not any(traj_classes):
            incorrect_data.append(item)
        else:
            for idx, is_correct in enumerate(traj_classes):
                single = {
                    'question': question,
                    'gold_answer': gold_answer,
                    'trajectories': [trajectories[idx]]
                }
                if is_correct:
                    correct_data.append(single)
                else:
                    incorrect_data.append(single)

    with open(correct_file, 'w', encoding='utf-8') as f:
        json.dump(correct_data, f, ensure_ascii=False, indent=2)

    with open(incorrect_file, 'w', encoding='utf-8') as f:
        json.dump(incorrect_data, f, ensure_ascii=False, indent=2)

    total_traj = correct_traj_count + incorrect_traj_count
    accuracy = correct_traj_count / total_traj if total_traj > 0 else 0.0

    return {
        "total_traj": total_traj,
        "correct_traj": correct_traj_count,
        "incorrect_traj": incorrect_traj_count,
        "accuracy": accuracy
    }


# --------- 多数据集批处理 ---------
if __name__ == "__main__":

    DATASETS = [
        {
            "name": "GSM8K",
            "input": "/hy-tmp/AutoTraj/eval_datasets/AutoTraj/gsm8k_test_autotraj.json",
            "correct": "/hy-tmp/AutoTraj/eval_datasets/AutoTraj/gsm8k_test_ac_autotraj.json",
            "incorrect": "/hy-tmp/AutoTraj/eval_datasets/AutoTraj/gsm8k_test_aw_autotraj.json",
        },
        {
            "name": "AIME24",
            "input": "/hy-tmp/AutoTraj/eval_datasets/AutoTraj/aime24_autotraj.json",
            "correct": "/hy-tmp/AutoTraj/eval_datasets/AutoTraj/aime24_ac_autotraj.json",
            "incorrect": "/hy-tmp/AutoTraj/eval_datasets/AutoTraj/aime24_aw_autotraj.json",
        },
        {
            "name": "AIME25",
            "input": "/hy-tmp/AutoTraj/eval_datasets/AutoTraj/aime25_autotraj.json",
            "correct": "/hy-tmp/AutoTraj/eval_datasets/AutoTraj/aime25_ac_autotraj.json",
            "incorrect": "/hy-tmp/AutoTraj/eval_datasets/AutoTraj/aime25_aw_autotraj.json",
        },
        {
            "name": "math500",
            "input": "/hy-tmp/AutoTraj/eval_datasets/AutoTraj/math500_autotraj.json",
            "correct": "/hy-tmp/AutoTraj/eval_datasets/AutoTraj/math500_ac_autotraj.json",
            "incorrect": "/hy-tmp/AutoTraj/eval_datasets/AutoTraj/math500_aw_autotraj.json",
        },
        {
            "name": "amc23",
            "input": "/hy-tmp/AutoTraj/eval_datasets/AutoTraj/amc23_autotraj.json",
            "correct": "/hy-tmp/AutoTraj/eval_datasets/AutoTraj/amc23_ac_autotraj.json",
            "incorrect": "/hy-tmp/AutoTraj/eval_datasets/AutoTraj/amc23_aw_autotraj.json",
        },
        {
            "name": "2wiki",
            "input": "/hy-tmp/AutoTraj/eval_datasets/AutoTraj/2wiki_autotraj.json",
            "correct": "/hy-tmp/AutoTraj/eval_datasets/AutoTraj/2wiki_ac_autotraj.json",
            "incorrect": "/hy-tmp/AutoTraj/eval_datasets/AutoTraj/2wiki_aw_autotraj.json",
        },
        {
            "name": "hotpotqa",
            "input": "/hy-tmp/AutoTraj/eval_datasets/AutoTraj/hotpotqa_autotraj.json",
            "correct": "/hy-tmp/AutoTraj/eval_datasets/AutoTraj/hotpotqa_ac_autotraj.json",
            "incorrect": "/hy-tmp/AutoTraj/eval_datasets/AutoTraj/hotpotqa_aw_autotraj.json",
        },
        {
            "name": "musique",
            "input": "/hy-tmp/AutoTraj/eval_datasets/AutoTraj/musique_autotraj.json",
            "correct": "/hy-tmp/AutoTraj/eval_datasets/AutoTraj/musique_ac_autotraj.json",
            "incorrect": "/hy-tmp/AutoTraj/eval_datasets/AutoTraj/musique_aw_autotraj.json",
        },
        {
            "name": "bamboogle",
            "input": "/hy-tmp/AutoTraj/eval_datasets/AutoTraj/bamboogle_autotraj.json",
            "correct": "/hy-tmp/AutoTraj/eval_datasets/AutoTraj/bamboogle_ac_autotraj.json",
            "incorrect": "/hy-tmp/AutoTraj/eval_datasets/AutoTraj/bamboogle_aw_autotraj.json",
        },
        {
            "name": "math",
            "input": "/hy-tmp/AutoTraj/eval_datasets/AutoTraj/math_autotraj.json",
            "correct": "/hy-tmp/AutoTraj/eval_datasets/AutoTraj/math_ac_autotraj.json",
            "incorrect": "/hy-tmp/AutoTraj/eval_datasets/AutoTraj/math_aw_autotraj.json",
        },
        {
            "name": "hle",
            "input": "/hy-tmp/AutoTraj/eval_datasets/AutoTraj/hle_autotraj.json",
            "correct": "/hy-tmp/AutoTraj/eval_datasets/AutoTraj/hle_ac_autotraj.json",
            "incorrect": "/hy-tmp/AutoTraj/eval_datasets/AutoTraj/hle_aw_autotraj.json",
        },
        # 👉 继续往这里加
    ]

    print("\n📊 Trajectory Classification Summary\n" + "-" * 50)

    for cfg in DATASETS:
        print(f"\n🚀 Processing {cfg['name']}")

        stats = classify_trajectories(
            input_file=cfg["input"],
            correct_file=cfg["correct"],
            incorrect_file=cfg["incorrect"]
        )

        print(
            f"✅ {cfg['name']} | "
            f"Total: {stats['total_traj']} | "
            f"Correct: {stats['correct_traj']} | "
            f"Wrong: {stats['incorrect_traj']} | "
            f"Accuracy: {stats['accuracy']:.4f}"
        )
