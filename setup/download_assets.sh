#!/bin/bash

# 1. 경로 설정 (Bench 폴더 내부 기준)
PROJECT_ROOT=$(pwd)
REPO_PATH="$PROJECT_ROOT/HuatuoGPT-Vision"
DATA_DIR="$REPO_PATH/data"

# 2. 기존의 꼬인 데이터 싹 삭제 (Clean Start)
rm -rf "$DATA_DIR"
mkdir -p "$DATA_DIR/slake/imgs"
mkdir -p "$DATA_DIR/vqarad/imgs"

# 3. 필수 도구 설치
apt-get update && apt-get install -y unzip wget
pip install datasets tqdm pillow

# 4. SLAKE 처리 (폴더 구조 유지하며 압축 해제)
echo "🚀 [1/2] SLAKE 데이터셋 정밀 복구 시작..."
SLAKE_DIR="$DATA_DIR/slake"
wget -O "$SLAKE_DIR/imgs.zip" https://huggingface.co/datasets/BoKelvin/SLAKE/resolve/main/imgs.zip
wget -O "$SLAKE_DIR/raw_test.json" https://huggingface.co/datasets/BoKelvin/SLAKE/resolve/main/test.json

# [중요] -j 옵션을 제거하여 xmlab1/source.jpg 같은 구조를 유지합니다.
unzip -o "$SLAKE_DIR/imgs.zip" -d "$SLAKE_DIR/imgs"
rm "$SLAKE_DIR/imgs.zip"

python3 -c "
import json, os
with open('$SLAKE_DIR/raw_test.json', 'r') as f:
    data = json.load(f)
formatted = []
for item in data:
    if item.get('q_lang') == 'en':
        item['image_name'] = item['img_name'] # 'xmlab1/source.jpg' 구조 그대로 유지
        formatted.append(item)
with open('$SLAKE_DIR/test.json', 'w') as f:
    json.dump(formatted, f, indent=4)
"
echo "✅ SLAKE 복구 완료!"

# 5. VQA-RAD 처리
echo "🚀 [2/2] VQA-RAD 데이터셋 처리 중..."
python3 -c "
import os, json
from datasets import load_dataset
from tqdm import tqdm
target_path = '$DATA_DIR/vqarad'
ds = load_dataset('flaviagiammarino/vqa-rad', split='test')
formatted = []
for i, item in enumerate(tqdm(ds)):
    img_name = f'vqarad_test_{i}.jpg'
    item['image'].convert('RGB').save(os.path.join(target_path, 'imgs', img_name))
    formatted.append({'qid': f'vqarad_{i}', 'image_name': img_name, 'question': item['question'], 'answer': str(item['answer']), 'answer_type': item.get('answer_type', 'OPEN')})
with open(os.path.join(target_path, 'test.json'), 'w') as f:
    json.dump(formatted, f, indent=4)
"
echo "🎉 데이터 준비 끝!"