import os
import json
import csv
import re
import sys
import torch
from tqdm import tqdm

# =====================================================================
# 1. 경로 설정 (절대 경로)
# =====================================================================
repo_path = "/workspace/HuatuoGPT-Vision-Bench/HuatuoGPT-Vision"
project_root = "/workspace/HuatuoGPT-Vision-Bench"

if repo_path not in sys.path:
    sys.path.append(repo_path)

try:
    from cli import HuatuoChatbot, IMAGE_TOKEN_INDEX
    print(f"✅ [SLAKE] 모델 로드 성공: {repo_path}")
except ImportError:
    print(f"❌ 에러: {repo_path}에서 모듈을 찾을 수 없습니다.")
    sys.exit(1)

# =====================================================================
# 2. 채점 함수
# =====================================================================
def normalize_answer(text):
    if not text: return ""
    return re.sub(r'[^\w\s]', '', str(text).lower().strip()).strip()

def check_correctness(gt, pred, q_type):
    pred_norm = normalize_answer(pred)
    gt_norm = normalize_answer(gt)
    if q_type.upper() == "CLOSED":
        return 1.0 if (gt_norm in pred_norm or pred_norm in gt_norm) else 0.0
    return 1.0 if gt_norm in pred_norm else 0.0

# =====================================================================
# 3. 모델 및 데이터 설정
# =====================================================================
bot = HuatuoChatbot("FreedomIntelligence/HuatuoGPT-Vision-7b")
model, tokenizer = bot.model, bot.tokenizer
if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

BATCH_SIZE = 4
data_path = os.path.join(repo_path, "data", "slake", "test.json")
img_dir = os.path.join(repo_path, "data", "slake", "imgs")
output_csv = os.path.join(project_root, "results", "huatuo_slake_eval.csv")

os.makedirs(os.path.dirname(output_csv), exist_ok=True)
with open(output_csv, mode='w', encoding='utf-8', newline='') as f:
    csv.writer(f).writerow(["qid", "question", "q_type", "ground_truth", "pure_pred", "score"])

with open(data_path, 'r') as f:
    test_data = json.load(f)

# =====================================================================
# 4. 추론 루프
# =====================================================================
print(f"🚀 SLAKE 추론 시작 (총 {len(test_data)} 문항)...")

for i in tqdm(range(0, len(test_data), BATCH_SIZE)):
    batch = test_data[i : i + BATCH_SIZE]
    input_ids_list, image_paths, valid_items = [], [], []
    
    for item in batch:
        img_path = os.path.join(img_dir, item['image_name'])
        if not os.path.exists(img_path): continue
            
        # [수정 1] 인자 개수 맞춤
        prompt = bot.insert_image_placeholder(item['question'], 1)
        
        # [수정 2] cli.py의 함수 시그니처에 맞게 호출 (tokenizer 인자 제거)
        ids = bot.tokenizer_image_token(prompt, return_tensors='pt')
            
        if ids.dim() == 2: ids = ids.squeeze(0)
        input_ids_list.append(ids)