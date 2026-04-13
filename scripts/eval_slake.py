import os
import json
import csv
import re
import sys
import torch
from tqdm import tqdm

# =====================================================================
# 1. 경로 설정 (HuatuoGPT-Vision-Bench 내부 구조 반영)
# =====================================================================
current_dir = os.path.dirname(os.path.abspath(__file__)) # scripts/
project_root = os.path.abspath(os.path.join(current_dir, "..")) # Bench/
# 이제 내 폴더(Bench) 바로 아래에 있는 Vision 폴더를 참조합니다.
repo_path = os.path.join(project_root, "HuatuoGPT-Vision")

if repo_path not in sys.path:
    sys.path.append(repo_path)

try:
    from cli import HuatuoChatbot, IMAGE_TOKEN_INDEX
    print(f"✅ [SLAKE] 모듈 로드 성공: {repo_path}")
except ImportError:
    print(f"❌ [SLAKE] 에러: {repo_path}에서 cli.py를 찾을 수 없습니다.")
    sys.exit(1)

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
    return re.sub(r'[^\w\s]', '', str(text).lower().strip()).strip()

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
    csv.writer(f).writerow(["qid", "question", "q_type", "ground_truth", "pure_pred", "preprocessed_pred", "score"])

with open(data_path, 'r') as f:
    test_data = json.load(f)

# =====================================================================
# 4. 추론 루프
# =====================================================================
for i in tqdm(range(0, len(test_data), BATCH_SIZE)):
    batch = test_data[i : i + BATCH_SIZE]
    try:
        input_ids_list, image_paths, valid_items = [], [], []
        for item in batch:
            img_path = os.path.join(img_dir, item['image_name'])
            if not os.path.exists(img_path): continue
            prompt = bot.insert_image_placeholder(item['question'])
            try: ids = bot.tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            except:
                from cli import tokenizer_image_token
                ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
            if ids.dim() == 2: ids = ids.squeeze(0)
            input_ids_list.append(ids); image_paths.append(img_path); valid_items.append(item)

        if not valid_items: continue
        img_tensors = bot.get_image_tensors(image_paths)
        if isinstance(img_tensors, list): img_tensors = torch.cat(img_tensors, dim=0)
        img_tensors = img_tensors.half().cuda()
        
        max_len = max([len(ids) for ids in input_ids_list])
        padded_ids, attention_masks = [], []
        for ids in input_ids_list:
            pad_len = max_len - len(ids)
            padded_ids.append(torch.cat([torch.full((pad_len,), tokenizer.pad_token_id, dtype=ids.dtype), ids]))
            attention_masks.append(torch.cat([torch.zeros(pad_len, dtype=torch.long), torch.ones(len(ids), dtype=torch.long)]))

        input_ids_tensor = torch.stack(padded_ids).cuda()
        attention_mask_tensor = torch.stack(attention_masks).cuda()
        
        with torch.no_grad():
            output_ids = model.generate(input_ids=input_ids_tensor, attention_mask=attention_mask_tensor, images=img_tensors, do_sample=False, max_new_tokens=128, use_cache=True)
        responses = tokenizer.batch_decode(output_ids[:, input_ids_tensor.shape[1]:], skip_special_tokens=True)
        
        with open(output_csv, mode='a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            for item, response in zip(valid_items, responses):
                writer.writerow([item['qid'], item['question'], item.get('answer_type', 'OPEN'), item['answer'], response.strip(), normalize_answer(response), check_correctness(item['answer'], response, item.get('answer_type', 'OPEN'))])
    except Exception:
        torch.cuda.empty_cache(); continue

print(f"🎉 SLAKE 완료! {output_csv}")