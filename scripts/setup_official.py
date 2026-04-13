import urllib.request
import json
import os

print("🚀 공식 JSON 파일 로드 및 분석 중...")
url = "https://huggingface.co/datasets/FreedomIntelligence/Medical_Multimodal_Evaluation_Data/resolve/main/medical_multimodel_evaluation_data.json"
official_json_path = "/workspace/HuatuoGPT-Vision-Bench/HuatuoGPT-Vision/data/official_data.json"

if not os.path.exists(official_json_path):
    print("👉 JSON 파일을 다운로드합니다 (약 5MB)...")
    urllib.request.urlretrieve(url, official_json_path)

with open(official_json_path, 'r', encoding='utf-8') as f:
    official_data = json.load(f)

# VQA-RAD는 파일명이 다를 수 있어 질문 텍스트를 기준으로 기존 이미지(vqarad_test_x.jpg)를 매핑합니다.
vqarad_local_path = "/workspace/HuatuoGPT-Vision-Bench/HuatuoGPT-Vision/data/vqarad/test.json"
vqarad_q_to_img = {}
if os.path.exists(vqarad_local_path):
    with open(vqarad_local_path, 'r', encoding='utf-8') as f:
        local_vqarad = json.load(f)
        for item in local_vqarad:
            vqarad_q_to_img[item['question'].strip().lower()] = item['image_name']

slake_closed, vqarad_closed = [], []

for item in official_data:
    ds_name = item.get('dataset', item.get('source', '')).lower()
    
    q, a = item.get('question', ''), item.get('answer', '')
    if not q and 'conversations' in item:
        for conv in item['conversations']:
            if conv['from'] == 'human': q = conv['value'].replace('<image>\n', '').replace('\n<image>', '').strip()
            elif conv['from'] == 'gpt': a = conv['value'].strip()
            
    # [핵심 1] image 필드가 리스트([])로 되어 있는 경우를 안전하게 텍스트로 변환
    img_field = item.get('image', '')
    if isinstance(img_field, list) and len(img_field) > 0:
        img_path_str = str(img_field[0])
    else:
        img_path_str = str(img_field)
            
    # [핵심 2] 오직 CLOSED (Yes/No) 질문만 취급합니다.
    if a.strip().lower() not in ['yes', 'no']:
        continue
        
    if 'slake' in ds_name:
        # 경로명 추출 안전 보장
        local_img_name = img_path_str.split('slake/')[-1] if 'slake/' in img_path_str else img_path_str.split('/')[-1]
        slake_closed.append({'qid': item.get('id', ''), 'question': q, 'answer': a, 'image_name': local_img_name})
        
    elif 'vqa-rad' in ds_name or 'vqarad' in ds_name:
        local_img_name = vqarad_q_to_img.get(q.strip().lower())
        if local_img_name:
            vqarad_closed.append({'qid': item.get('id', ''), 'question': q, 'answer': a, 'image_name': local_img_name})

base_dir = "/workspace/HuatuoGPT-Vision-Bench/HuatuoGPT-Vision/data"
with open(f"{base_dir}/slake_official_closed.json", 'w', encoding='utf-8') as f: json.dump(slake_closed, f, indent=4)
with open(f"{base_dir}/vqarad_official_closed.json", 'w', encoding='utf-8') as f: json.dump(vqarad_closed, f, indent=4)

print(f"✅ 추출 완료! 기존 이미지와 연결되었습니다.")
print(f"👉 SLAKE CLOSED: {len(slake_closed)}개")
print(f"👉 VQA-RAD CLOSED: {len(vqarad_closed)}개")