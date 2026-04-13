#!/bin/bash

# 1. 경로 설정
PROJECT_ROOT=$(pwd)
REPO_NAME="HuatuoGPT-Vision"
REPO_PATH="$PROJECT_ROOT/$REPO_NAME"
DATA_DIR="$REPO_PATH/data"

echo "📂 작업 경로: $PROJECT_ROOT"
echo "📂 데이터 저장 경로: $DATA_DIR"

# 2. 필수 도구 설치 (unzip, wget)
apt-get update && apt-get install -y unzip wget
pip install datasets tqdm pillow

# 3. 원본 레포지토리 클론
if [ ! -d "$REPO_PATH" ]; then
    echo "🚀 [1/3] 원본 레포지토리 클론 중..."
    git clone https://github.com/FreedomIntelligence/HuatuoGPT-Vision.git "$REPO_PATH"
fi

# 4. SLAKE 데이터셋 처리 (직접 다운로드 및 압축 해제)
echo "🚀 [2/3] SLAKE 데이터셋 수동 다운로드 및 압축 해제..."
SLAKE_DIR="$DATA_DIR/slake"
mkdir -p "$SLAKE_DIR/imgs"

# 이미지 압축파일 및 JSON 직접 다운로드
wget -O "$SLAKE_DIR/imgs.zip" https://huggingface.co/datasets/BoKelvin/SLAKE/resolve/main/imgs.zip
wget -O "$SLAKE_DIR/raw_test.json" https://huggingface.co/datasets/BoKelvin/SLAKE/resolve/main/test.json

# 압축 해제 (imgs 폴더 안에 이미지들을 풉니다)
unzip -j -o "$SLAKE_DIR/imgs.zip" -d "$SLAKE_DIR/imgs"
rm "$SLAKE_DIR/imgs.zip"

# SLAKE 전처리 (영어 데이터 필터링 및 경로 최적화)
python3 -c "
import json, os
slake_dir = '$SLAKE_DIR'
with open(os.path.join(slake_dir, 'raw_test.json'), 'r', encoding='utf-8') as f:
    data = json.load(f)

# 영어 질문만 필터링하고 image_name 필드를 우리 규격에 맞게 조정
formatted = []
for item in data:
    if item.get('q_lang') == 'en':
        item['image_name'] = item['img_name']  # SLAKE 원본은 img_name 사용
        formatted.append(item)

with open(os.path.join(slake_dir, 'test.json'), 'w', encoding='utf-8') as f:
    json.dump(formatted, f, indent=4)
print(f'✅ SLAKE 전처리 완료: {len(formatted)}개 영어 데이터 확보')
"

# 5. VQA-RAD 처리 (기존 방식 유지 - 얘는 보통 잘 로드됩니다)
echo "🚀 [3/3] VQA-RAD 데이터셋 처리 중..."
python3 -c "
import os, json
from datasets import load_dataset
from tqdm import tqdm

target_path = os.path.join('$DATA_DIR', 'vqarad')
img_dir = os.path.join(target_path, 'imgs')
os.makedirs(img_dir, exist_ok=True)

try:
    ds = load_dataset('flaviagiammarino/vqa-rad', split='test')
    formatted = []
    for i, item in enumerate(tqdm(ds)):
        img_name = f'vqarad_test_{i}.jpg'
        item['image'].convert('RGB').save(os.path.join(img_dir, img_name))
        formatted.append({
            'qid': f'vqarad_{i}',
            'image_name': img_name,
            'question': item['question'],
            'answer': str(item['answer']),
            'answer_type': item.get('answer_type', 'CLOSED' if str(item['answer']).lower() in ['yes', 'no'] else 'OPEN')
        })
    with open(os.path.join(target_path, 'test.json'), 'w', encoding='utf-8') as f:
        json.dump(formatted, f, indent=4)
    print('✅ VQA-RAD 완료!')
except Exception as e:
    print(f'❌ VQA-RAD 에러: {e}')
"

echo "🎉 모든 데이터 준비가 완료되었습니다! 이제 평가를 시작하세요."