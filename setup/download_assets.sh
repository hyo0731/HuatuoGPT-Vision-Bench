#!/bin/bash

# 1. 경로 설정 (절대 경로 고정)
PROJECT_ROOT="/workspace/HuatuoGPT-Vision-Bench"
REPO_PATH="$PROJECT_ROOT/HuatuoGPT-Vision"
DATA_DIR="$REPO_PATH/data"

echo "📂 작업 경로: $PROJECT_ROOT"

# 2. 원본 레포지토리 클론
if [ ! -d "$REPO_PATH" ]; then
    echo "🚀 [1/3] 원본 레포지토리 클론 중..."
    git clone https://github.com/FreedomIntelligence/HuatuoGPT-Vision.git "$REPO_PATH"
fi

# 3. 필수 도구 설치
apt-get update && apt-get install -y unzip wget
pip install datasets tqdm pillow setuptools

# 4. SLAKE 데이터셋 처리 (중복 폴더 방지 로직)
echo "🚀 [2/3] SLAKE 데이터셋 수동 다운로드 및 압축 해제..."
SLAKE_DIR="$DATA_DIR/slake"
rm -rf "$SLAKE_DIR" # 깨끗하게 시작
mkdir -p "$SLAKE_DIR"

wget -O "$SLAKE_DIR/imgs.zip" https://huggingface.co/datasets/BoKelvin/SLAKE/resolve/main/imgs.zip
wget -O "$SLAKE_DIR/raw_test.json" https://huggingface.co/datasets/BoKelvin/SLAKE/resolve/main/test.json

# 압축 해제 (SLAKE_DIR 폴더에 바로 풉니다)
unzip -o "$SLAKE_DIR/imgs.zip" -d "$SLAKE_DIR"
rm "$SLAKE_DIR/imgs.zip"

# [핵심] 만약 imgs/imgs/ 구조로 풀렸다면 밖으로 끄집어내기
if [ -d "$SLAKE_DIR/imgs/imgs" ]; then
    mv "$SLAKE_DIR/imgs/imgs"/* "$SLAKE_DIR/imgs/"
    rm -rf "$SLAKE_DIR/imgs/imgs"
fi

# 불필요한 맥 OS용 파일 삭제
rm -rf "$SLAKE_DIR/__MACOSX" "$SLAKE_DIR/imgs/__MACOSX"

python3 -c "
import json, os
with open('$SLAKE_DIR/raw_test.json', 'r') as f:
    data = json.load(f)
formatted = []
for item in data:
    if item.get('q_lang') == 'en':
        item['image_name'] = item['img_name']
        formatted.append(item)
with open('$SLAKE_DIR/test.json', 'w') as f:
    json.dump(formatted, f, indent=4)
print(f'✅ SLAKE 전처리 완료: {len(formatted)}개 데이터 확보')
"

# 5. VQA-RAD 처리
echo "🚀 [3/3] VQA-RAD 데이터셋 처리 중..."
python3 -c "
import os, json
from datasets import load_dataset
from tqdm import tqdm
target_path = '$DATA_DIR/vqarad'
img_dir = os.path.join(target_path, 'imgs')
os.makedirs(img_dir, exist_ok=True)
ds = load_dataset('flaviagiammarino/vqa-rad', split='test')
formatted = []
for i, item in enumerate(tqdm(ds)):
    img_name = f'vqarad_test_{i}.jpg'
    item['image'].convert('RGB').save(os.path.join(img_dir, img_name))
    formatted.append({'qid': f'vqarad_{i}', 'image_name': img_name, 'question': item['question'], 'answer': str(item['answer']), 'answer_type': item.get('answer_type', 'OPEN')})
with open(os.path.join(target_path, 'test.json'), 'w') as f:
    json.dump(formatted, f, indent=4)
"
echo "🎉 모든 데이터 준비 완료!"