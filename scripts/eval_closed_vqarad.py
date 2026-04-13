import os
import json
import csv
import re
import sys
from tqdm import tqdm

# =====================================================================
# 1. 평가 대상 (slake 또는 vqarad)
# =====================================================================
TARGET_DATASET = 'vqarad'  

# 경로 설정
repo_path = "/workspace/HuatuoGPT-Vision-Bench/HuatuoGPT-Vision"
if repo_path not in sys.path:
    sys.path.append(repo_path)
from cli import HuatuoChatbot

# =====================================================================
# 2. 초간단 CLOSED 채점 함수
# =====================================================================
def normalize_answer(text):
    if not text: return ""
    return re.sub(r'[^\w\s]', '', str(text).lower().strip()).strip()

def check_correctness_closed(gt, pred):
    pred_norm = normalize_answer(pred)
    gt_norm = normalize_answer(gt)
    return 1.0 if (gt_norm in pred_norm or pred_norm in gt_norm) else 0.0

# =====================================================================
# 3. 데이터 로드 및 모델 준비
# =====================================================================
print(f"🤖 {TARGET_DATASET.upper()} CLOSED 평가 모델 로딩 중...")
bot = HuatuoChatbot("FreedomIntelligence/HuatuoGPT-Vision-7b")

data_path = f"{repo_path}/data/{TARGET_DATASET}_official_closed.json"
img_dir = f"{repo_path}/data/{TARGET_DATASET}/imgs"
output_csv = f"/workspace/HuatuoGPT-Vision-Bench/results/official_closed_{TARGET_DATASET}.csv"

os.makedirs(os.path.dirname(output_csv), exist_ok=True)
with open(output_csv, mode='w', encoding='utf-8', newline='') as f:
    csv.writer(f).writerow(["qid", "question", "ground_truth", "prediction", "score"])

with open(data_path, 'r', encoding='utf-8') as f:
    test_data = json.load(f)

print(f"\n🚀 총 {len(test_data)}개 {TARGET_DATASET.upper()} 평가 시작...")

# =====================================================================
# 4. 순차 추론 
# =====================================================================
for i, item in enumerate(tqdm(test_data)):
    # [핵심 수정] JSON에 남아있는 'imgs/' 접두사를 제거하고 정확한 경로 조립
    clean_img_name = item['image_name'].replace('imgs/', '')
    image_path = os.path.join(img_dir, clean_img_name)
    
    if not os.path.exists(image_path):
        if i == 0: 
            print(f"\n⚠️ 경로 에러! 파이썬이 찾으려던 경로: {image_path}")
        continue
    
    try:
        bot.clear_history()
        
        # 추론 시도
        try:
            pure_pred_list = bot.inference(item['question'], [image_path])
            pure_pred = pure_pred_list[0] if pure_pred_list else "" 
        except TypeError:
            pure_pred = bot.chat(item['question'], [image_path])
        except Exception as e:
            print(f"\n⚠️ 모델 추론 에러 (QID: {item['qid']}): {e}")
            continue
        
        # 채점
        score = check_correctness_closed(item['answer'], pure_pred)
        
        # 저장
        with open(output_csv, mode='a', encoding='utf-8', newline='') as f:
            csv.writer(f).writerow([item['qid'], item['question'], item['answer'], pure_pred, score])
            
    except Exception as e:
        print(f"\n⚠️ 알 수 없는 에러: {e}")
        continue

print(f"\n🎉 평가 완료! 결과: {output_csv}")