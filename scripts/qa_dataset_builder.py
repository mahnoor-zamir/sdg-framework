import json
import os

SDG_JSON_PATH = os.path.join('data', 'sdg_structured.json')
QA_PAIRS_PATH = os.path.join('data', 'qa_pairs.json')

with open(SDG_JSON_PATH, 'r', encoding='utf-8') as f:
    sdg_data = json.load(f)

qa_pairs = []

for goal_num, goal in sdg_data.items():
    goal_name = goal['name']
    goal_context = f"Goal {goal_num}: {goal_name}"
    # Goal-level questions
    qa_pairs.append({
        "question": f"What is SDG {goal_num}?",
        "context": goal_context,
        "answer": goal_name
    })
    qa_pairs.append({
        "question": f"List all targets for SDG {goal_num}.",
        "context": goal_context,
        "answer": ", ".join(goal['targets'].keys())
    })
    for target_code, target in goal['targets'].items():
        target_context = f"{goal_context} Target {target_code}: {target['description']}"
        # Target-level questions
        qa_pairs.append({
            "question": f"What is Target {target_code} of SDG {goal_num}?",
            "context": target_context,
            "answer": target['description']
        })
        qa_pairs.append({
            "question": f"List all indicators for Target {target_code} of SDG {goal_num}.",
            "context": target_context,
            "answer": ", ".join([ind['code'] for ind in target['indicators']])
        })
        for ind in target['indicators']:
            ind_context = f"{target_context} Indicator {ind['code']}: {ind['description']}"
            # Indicator-level questions
            qa_pairs.append({
                "question": f"What does Indicator {ind['code']} measure?",
                "context": ind_context,
                "answer": ind['description']
            })
            qa_pairs.append({
                "question": f"Which SDG target does Indicator {ind['code']} belong to?",
                "context": ind_context,
                "answer": target_code
            })

with open(QA_PAIRS_PATH, 'w', encoding='utf-8') as f:
    json.dump(qa_pairs, f, indent=2, ensure_ascii=False)

print(f"Generated {len(qa_pairs)} QA pairs and saved to {QA_PAIRS_PATH}.")
print("Please review and augment the dataset with more diverse and complex questions as needed.") 