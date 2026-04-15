import os
import json
import csv
import re
import sys
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# =====================================================================
# 1. 경로 설정
# =====================================================================
repo_path = "/workspace/HuatuoGPT-Vision-Bench/HuatuoGPT-Vision"
project_root = "/workspace/HuatuoGPT-Vision-Bench"

if repo_path not in sys.path:
    sys.path.append(repo_path)

from cli import HuatuoChatbot

# =====================================================================
# 2. 이미지 변형 함수 (Perturbations)
# =====================================================================
def apply_fft_lpf(img_path, keep_ratio):
    """저주파 통과 필터: 형체만 남기고 디테일 삭제"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    rows, cols = img.shape
    crow, ccol = rows//2, cols//2
    r = int(min(rows, cols) * keep_ratio)
    mask = np.zeros((rows, cols), np.uint8)
    if r > 0:
        mask[crow-r:crow+r, ccol-r:ccol+r] = 1 
    else:
        mask[crow, ccol] = 1 
    fshift = fshift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.abs(np.fft.ifft2(f_ishift))
    
    # [핵심 수정] LPF는 밝기가 유지되므로 0~255 밖으로 튀어나온 값만 잘라냅니다.
    img_back = np.clip(img_back, 0, 255).astype(np.uint8)
    return img_back

def apply_fft_hpf(img_path, cut_ratio):
    """고주파 통과 필터: 형체를 지우고 윤곽선(Edge)만 남김"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    rows, cols = img.shape
    crow, ccol = rows//2, cols//2
    r = int(min(rows, cols) * cut_ratio)
    mask = np.ones((rows, cols), np.uint8)
    if r > 0:
        mask[crow-r:crow+r, ccol-r:ccol+r] = 0 
    fshift = fshift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.abs(np.fft.ifft2(f_ishift))
    
    # [핵심 수정] HPF는 전체 밝기가 날아갔으므로, 
    # 가장 어두운 곳을 0, 가장 밝은 윤곽선을 255로 쫙 늘려줍니다 (정규화)
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    img_back = np.uint8(img_back)
    return img_back

def apply_sp_noise(img_path, prob):
    """Salt and Pepper 노이즈 추가"""
    img = cv2.imread(img_path)
    if img is None: return None
    noise = np.random.rand(*img.shape[:2])
    img[noise < (prob / 2)] = 0
    img[noise > 1 - (prob / 2)] = 255
    return img

# =====================================================================
# 3. 시각화 함수 (다중 조건 한눈에 보기)
# =====================================================================
def visualize_samples(test_data, img_dir, save_path, num_samples=3):
    print(f"\n👀 이미지 변형 샘플 {num_samples}개 시각화 중...")
    
    # 텍스트 조건을 제외한 시각적 변형 조건들
    vis_conds = [
        ("Original", None, None),
        ("LPF 20%", apply_fft_lpf, 0.20),
        ("LPF 10%", apply_fft_lpf, 0.10),
        ("LPF 5%", apply_fft_lpf, 0.05),
        ("LPF 1%", apply_fft_lpf, 0.01),
        ("HPF 5%", apply_fft_hpf, 0.05),
        ("HPF 10%", apply_fft_hpf, 0.10),
        ("Noise 5%", apply_sp_noise, 0.05),
        ("Noise 10%", apply_sp_noise, 0.10)
    ]
    
    num_cols = len(vis_conds)
    fig, axes = plt.subplots(num_samples, num_cols, figsize=(3 * num_cols, 3 * num_samples))
    
    # 제목 설정
    for col_idx, (title, _, _) in enumerate(vis_conds):
        axes[0, col_idx].set_title(title, fontsize=12)
    
    sample_idx = 0
    for item in test_data:
        if sample_idx >= num_samples: break
        
        orig_img_path = os.path.join(img_dir, item['image_name'])
        if not os.path.exists(orig_img_path): continue
        orig_img = cv2.imread(orig_img_path)
        if orig_img is None: continue
            
        for col_idx, (title, func, param) in enumerate(vis_conds):
            ax = axes[sample_idx, col_idx]
            ax.axis('off')
            
            if func is None:
                ax.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
            else:
                mod_img = func(orig_img_path, param)
                if mod_img is not None:
                    # LPF, HPF는 흑백이므로 컬러맵 적용
                    if "LPF" in title or "HPF" in title:
                        ax.imshow(mod_img, cmap='gray')
                    else:
                        ax.imshow(cv2.cvtColor(mod_img, cv2.COLOR_BGR2RGB))
                        
        sample_idx += 1
        
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ 시각화 완료! 파일 저장됨: {save_path}")

# =====================================================================
# 4. 채점 및 분류 헬퍼
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

def categorize_question(q):
    q_lower = q.lower()
    if re.search(r'\b(is there|are there|do you see)\b', q_lower): return "Presence"
    if re.search(r'\b(enlarged|size|measure|how long|ratio)\b', q_lower): return "Quantitative"
    if re.search(r'\b(appear|level|slice|cut|boundary)\b', q_lower): return "Spatial/Slice"
    if re.search(r'\b(neoplastic|abnormal|tumor|cancer|mass|lesion)\b', q_lower): return "Abnormality"
    return "General"

# =====================================================================
# 5. PyTorch Dataset 구성
# =====================================================================
class MedVLMExperimentDataset(Dataset):
    def __init__(self, json_data, img_dir, temp_dir):
        self.img_dir = img_dir
        self.temp_dir = temp_dir
        os.makedirs(self.temp_dir, exist_ok=True)
        
        self.flat_data = []
        # 총 10가지 조건 리스트업
        self.conditions = [
            "Baseline", 
            "LPF_20", "LPF_10", "LPF_05", "LPF_01",
            "HPF_05", "HPF_10",
            "Noise_05", "Noise_10", 
            "No_Image"
        ]
        
        for item in json_data:
            orig_img_path = os.path.join(self.img_dir, item['image_name'])
            if not os.path.exists(orig_img_path): continue
            for cond in self.conditions:
                self.flat_data.append((item, cond, orig_img_path))

    def __len__(self):
        return len(self.flat_data)

    def __getitem__(self, idx):
        item, cond_name, orig_img_path = self.flat_data[idx]
        qid = item.get('qid', 'N/A')
        question = item['question']
        q_type = item.get('answer_type', 'OPEN')
        gt = item['answer']
        q_category = categorize_question(question)
        
        out_img_path = ""
        
        if cond_name != "No_Image":
            file_name = f"{cond_name}_{qid}_{os.path.basename(orig_img_path)}"
            out_img_path = os.path.join(self.temp_dir, file_name)
            
            if not os.path.exists(out_img_path):
                img_back = None
                if cond_name == "Baseline":
                    out_img_path = orig_img_path
                elif cond_name == "LPF_20": img_back = apply_fft_lpf(orig_img_path, 0.20)
                elif cond_name == "LPF_10": img_back = apply_fft_lpf(orig_img_path, 0.10)
                elif cond_name == "LPF_05": img_back = apply_fft_lpf(orig_img_path, 0.05)
                elif cond_name == "LPF_01": img_back = apply_fft_lpf(orig_img_path, 0.01)
                elif cond_name == "HPF_05": img_back = apply_fft_hpf(orig_img_path, 0.05)
                elif cond_name == "HPF_10": img_back = apply_fft_hpf(orig_img_path, 0.10)
                elif cond_name == "Noise_05": img_back = apply_sp_noise(orig_img_path, 0.05)
                elif cond_name == "Noise_10": img_back = apply_sp_noise(orig_img_path, 0.10)
                
                if img_back is not None:
                    cv2.imwrite(out_img_path, img_back)
                    
        return {
            "qid": qid,
            "question": question,
            "q_type": q_type,
            "q_category": q_category,
            "gt": gt,
            "cond_name": cond_name,
            "img_path": out_img_path
        }

def custom_collate(batch):
    return {key: [d[key] for d in batch] for key in batch[0]}

# =====================================================================
# 6. 메인 실행 루프
# =====================================================================
if __name__ == "__main__":
    # 데이터셋 경로를 vqarad로 수정
    data_path = os.path.join(repo_path, "data", "vqarad", "test.json")
    img_dir = os.path.join(repo_path, "data", "vqarad", "imgs")
    
    # 임시 폴더 및 결과 파일명도 vqarad용으로 수정
    temp_dir = os.path.join(project_root, "temp_exps_vqarad")
    output_csv = os.path.join(project_root, "results", "huatuo_vqarad_experiments_fast.csv")
    vis_path = os.path.join(project_root, "results", "vqarad_image_perturbations_sample.png")

    with open(data_path, 'r') as f:
        test_data = json.load(f)
        
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # 1. 시각화 확인 (너무 크지 않게 3장만 표시)
    visualize_samples(test_data, img_dir, vis_path, num_samples=3)

    # 2. 모델 평가
    print("\n🤖 VQA-RAD 모델 로딩 중...")
    bot = HuatuoChatbot("FreedomIntelligence/HuatuoGPT-Vision-7b")

    dataset = MedVLMExperimentDataset(test_data, img_dir, temp_dir)
    batch_size = 16  
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, collate_fn=custom_collate)

    headers = ["qid", "question", "q_type", "q_category", "ground_truth", "condition", "pure_pred", "score"]
    with open(output_csv, mode='w', encoding='utf-8', newline='') as f:
        csv.writer(f).writerow(headers)

    print(f"\n🚀 총 {len(dataset)}개 (샘플 x 10조건) 고속 평가 시작...")

    for batch in tqdm(dataloader):
        batch_results = []
        for i in range(len(batch["qid"])):
            qid = batch["qid"][i]
            question = batch["question"][i]
            q_type = batch["q_type"][i]
            q_category = batch["q_category"][i]
            gt = batch["gt"][i]
            cond_name = batch["cond_name"][i]
            img_path = batch["img_path"][i]
            
            bot.clear_history()
            pure_pred = ""
            
            try:
                if cond_name == "No_Image":
                    try:
                        pure_pred_list = bot.inference(question, [])
                        pure_pred = pure_pred_list[0] if pure_pred_list else ""
                    except Exception:
                        try:
                            pure_pred = bot.chat(question)
                        except Exception as e:
                            pure_pred = f"NO_IMAGE_ERROR: {str(e)}"
                else:
                    try:
                        pure_pred_list = bot.inference(question, [img_path])
                        pure_pred = pure_pred_list[0] if pure_pred_list else "" 
                    except TypeError:
                        pure_pred = bot.chat(question, [img_path])
            except Exception as e:
                pure_pred = f"ERROR: {str(e)}"
                
            score = check_correctness(gt, pure_pred, q_type)
            batch_results.append([qid, question, q_type, q_category, gt, cond_name, pure_pred, score])
                
        with open(output_csv, mode='a', encoding='utf-8', newline='') as f:
            csv.writer(f).writerows(batch_results)

    print(f"\n🎉 VQA-RAD 실험 완료! 평가결과는 {output_csv}, 시각화 이미지는 {vis_path}를 확인하세요.")