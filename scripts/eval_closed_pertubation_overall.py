import os
import json
import csv
import re
import sys
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from itertools import zip_longest
from numpy.fft import fft2, ifft2, fftshift, ifftshift

# =====================================================================
# 1. 경로 설정 및 모듈 임포트
# =====================================================================
repo_path = "/workspace/HuatuoGPT-Vision-Bench/HuatuoGPT-Vision"
if repo_path not in sys.path:
    sys.path.append(repo_path)
from cli import HuatuoChatbot

# =====================================================================
# 2. 이미지 변형 엔진 (White, Gray 추가 및 NumPy FFT 적용)
# =====================================================================
def apply_perturbation(image, p_type, val=0.5):
    img_color = np.array(image.convert('RGB'))
    h, w, _ = img_color.shape
    cy, cx = h // 2, w // 2

    # 단색 이미지 (Black, White, Gray)
    if p_type == 'black': return Image.new('RGB', image.size, (0, 0, 0))
    if p_type == 'white': return Image.new('RGB', image.size, (255, 255, 255))
    if p_type == 'gray': return Image.new('RGB', image.size, (128, 128, 128))

    # 패치 셔플 (Shuffle)
    if p_type == 'shuffle':
        patch_size = 24
        patches = []
        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                patches.append(img_color[i:i+patch_size, j:j+patch_size])
        np.random.shuffle(patches)
        shuffled_img = np.zeros_like(img_color)
        idx = 0
        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                if idx < len(patches):
                    shuffled_img[i:i+patch_size, j:j+patch_size] = patches[idx]
                    idx += 1
        return Image.fromarray(shuffled_img)

    # 주파수 필터링 (LPF, HPF)
    transformed_channels = []
    for c in range(3):
        f = fft2(img_color[:, :, c])
        fshift = fftshift(f)
        
        y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
        dist = np.sqrt(x*x + y*y)
        mask_radius = val * (min(h, w) / 2)

        if p_type == 'lpf':
            mask = dist <= mask_radius
        elif p_type == 'hpf':
            mask = dist > mask_radius
        else:
            mask = np.ones_like(dist, dtype=bool)
        
        fshift = fshift * mask
        f_ishift = ifftshift(fshift)
        img_back = ifft2(f_ishift)
        transformed_channels.append(np.abs(img_back))
    
    final_img = np.stack(transformed_channels, axis=2).clip(0, 255).astype(np.uint8)
    return Image.fromarray(final_img)

# =====================================================================
# 3. 채점 함수
# =====================================================================
def normalize_answer(text):
    if not text: return ""
    return re.sub(r'[^\w\s]', '', str(text).lower().strip()).strip()

def check_correctness_closed(gt, pred):
    return 1.0 if (normalize_answer(gt) in normalize_answer(pred) or normalize_answer(pred) in normalize_answer(gt)) else 0.0

# =====================================================================
# 4. 데이터 로드 및 셔플링 (SLAKE 1개 -> VQA-RAD 1개 교차)
# =====================================================================
print("📂 데이터셋 로딩 및 교차(Interleave) 병합 중...")

slake_path = f"{repo_path}/data/slake_official_closed.json"
vqarad_path = f"{repo_path}/data/vqarad_official_closed.json"

with open(slake_path, 'r') as f: slake_data = json.load(f)
with open(vqarad_path, 'r') as f: vqarad_data = json.load(f)

# [핵심] 번갈아가며 리스트에 담기
mixed_tasks = []
for s_item, v_item in zip_longest(slake_data, vqarad_data):
    if s_item: mixed_tasks.append(('slake', s_item))
    if v_item: mixed_tasks.append(('vqarad', v_item))

# =====================================================================
# 5. 실험 목록 및 모델 로딩
# =====================================================================
experiments = [
    ('lpf', 0.3), ('lpf', 0.5), ('lpf', 0.7),
    ('hpf', 0.3), ('hpf', 0.5), ('hpf', 0.7),
    ('shuffle', None), 
    ('black', None), ('white', None), ('gray', None) # 3가지 단색 추가
]

print("🤖 모델 로딩 중 (VRAM 할당)...")
bot = HuatuoChatbot("FreedomIntelligence/HuatuoGPT-Vision-7b")

# =====================================================================
# 6. 밤샘 메인 루프 (절대 죽지 않는 구조)
# =====================================================================
for p_type, p_val in experiments:
    exp_name = f"{p_type}_{p_val}" if p_val else p_type
    print(f"\n==================================================")
    print(f"🚀 현재 실험 진행 중: [ {exp_name.upper()} ] 변형")
    print(f"==================================================")
    
    csv_paths = {
        'slake': f"/workspace/HuatuoGPT-Vision-Bench/results/ablation_slake_{exp_name}.csv",
        'vqarad': f"/workspace/HuatuoGPT-Vision-Bench/results/ablation_vqarad_{exp_name}.csv"
    }

    # CSV 헤더 초기화
    for path in csv_paths.values():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            with open(path, mode='w', encoding='utf-8', newline='') as f:
                csv.writer(f).writerow(["qid", "question", "ground_truth", "prediction", "score"])

    # 교차 병합된 데이터 순회
    for dataset_name, item in tqdm(mixed_tasks, desc=f"Evaluating {exp_name}"):
        try:
            # 1. 경로 조립 (데이터셋별 차이 처리)
            img_dir = f"{repo_path}/data/{dataset_name}/imgs"
            clean_img_name = item['image_name'].replace('imgs/', '')
            
            if dataset_name == 'slake' and '_source.jpg' in clean_img_name:
                clean_img_name = clean_img_name.replace('_source.jpg', '/source.jpg')
                
            image_path = os.path.join(img_dir, clean_img_name)
            
            if not os.path.exists(image_path): continue # 파일 없으면 조용히 패스

            # 2. 이미지 변형 및 임시 저장
            raw_img = Image.open(image_path)
            pert_img = apply_perturbation(raw_img, p_type, p_val)
            temp_pert_path = f"/tmp/current_pert_{exp_name}.jpg" # 덮어쓰기 방식으로 저장 용량 절약
            pert_img.save(temp_pert_path)

            # 3. 모델 추론
            bot.clear_history()
            try:
                pure_pred_list = bot.inference(item['question'], [temp_pert_path])
                pure_pred = pure_pred_list[0] if pure_pred_list else ""
            except TypeError:
                pure_pred = bot.chat(item['question'], [temp_pert_path])
            
            # 4. 채점 및 즉시 파일 저장 (전원 꺼져도 중간 결과 보존)
            score = check_correctness_closed(item['answer'], pure_pred)
            
            with open(csv_paths[dataset_name], mode='a', encoding='utf-8', newline='') as f:
                csv.writer(f).writerow([item['qid'], item['question'], item['answer'], pure_pred, score])

        except Exception as e:
            # 에러가 나도 캐시만 비우고 다음 문제로 넘어감 (강제 종료 방지)
            torch.cuda.empty_cache()
            continue

print(f"\n🎉🎉 밤샘 자동화 실험 대성공! 모든 결과가 results/ 폴더에 저장되었습니다! 🎉🎉")