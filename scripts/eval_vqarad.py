import os
import json
import csv
import re
import sys
from tqdm import tqdm

# =====================================================================
# 1. 경로 설정 (HuatuoGPT-Vision-Bench 내부 구조 반영)
# =====================================================================
repo_path = "/workspace/HuatuoGPT-Vision-Bench/HuatuoGPT-Vision"
project_root = "/workspace/HuatuoGPT-Vision-Bench"

if repo_path not in sys.path:
    sys.path.append(repo_path)

from cli import HuatuoChatbot

# =====================================================================
# 2. 채점 함수
# =====================================================================
def get_word_variations(word):
    variations = {word}
    if word.endswith('ies'): variations.add(word[:-3] + 'y')
    elif word.endswith('es') and len(word) > 3: variations.update([word[:-2], word[:-1]])
    elif word.endswith('s') and len(word) > 2: variations.add(word[:-1])
    else: variations.update([word + 's', word + 'es'])
    return list(variations)

def normalize_answer(text):
    if not text: return ""
    text = str(text).lower().strip()
    return re.sub(r'[^\w\s]', '', text).strip()

def check_correctness(gt, pred, q_type):
    pred_norm = normalize_answer(pred)
    if q_type.upper() == "CLOSED":
        gt_norm = normalize_answer(gt)
        return 1.0 if (gt_norm in pred_norm or pred_norm in gt_norm) else 0.0
    else:
        gt_items = [item.strip() for item in str(gt).split(',')]
        if not gt_items or gt_items == ['']: return 0.0
        match_count = sum(1 for item in gt_items if normalize_answer(item) and re.search(r'\b(' + '|'.join(map(re.escape, get_word_variations(normalize_answer(item)))) + r')\b', pred_norm))
        return round(match_count / len(gt_items), 4)

# =====================================================================
# 3. 모델 준비
# =====================================================================
print("🤖 VQA-RAD 모델 로딩 중...")
bot = HuatuoChatbot("FreedomIntelligence/HuatuoGPT-Vision-7b")

data_path = os.path.join(repo_path, "data", "vqarad", "test.json")
img_dir = os.path.join(repo_path, "data", "vqarad", "imgs")
output_csv = os.path.join(project_root, "results", "huatuo_vqarad_eval.csv")

os.makedirs(os.path.dirname(output_csv), exist_ok=True)
headers = ["qid", "question", "q_type", "ground_truth", "pure_pred", "preprocessed_pred", "score"]
with open(output_csv, mode='w', encoding='utf-8', newline='') as f:
    csv.writer(f).writerow(headers)

with open(data_path, 'r') as f:
    test_data = json.load(f)

# =====================================================================
# 4. 안전한 순차 처리 (Sequential)
# =====================================================================
print(f"\n🚀 총 {len(test_data)}개 샘플 평가 시작... (결과는 실시간 CSV 저장)")

for item in tqdm(test_data):
    image_path = os.path.join(img_dir, item['image_name'])
    if not os.path.exists(image_path):
        continue
    
    qid = item.get('qid', 'N/A')
    question = item['question']
    q_type = item.get('answer_type', 'OPEN')
    gt = item['answer']
    
    try:
        bot.clear_history()
        
        try:
            pure_pred_list = bot.inference(question, [image_path])
            pure_pred = pure_pred_list[0] if pure_pred_list else "" 
        except TypeError:
            pure_pred = bot.chat(question, [image_path])
        except Exception as e:
            continue
        
        prep_pred = normalize_answer(pure_pred)
        score = check_correctness(gt, pure_pred, q_type)
        
        with open(output_csv, mode='a', encoding='utf-8', newline='') as f:
            csv.writer(f).writerow([qid, question, q_type, gt, pure_pred, prep_pred, score])
            
    except Exception as e:
        continue

print(f"\n🎉 VQA-RAD 평가 완료! {output_csv} 파일을 확인하세요.")