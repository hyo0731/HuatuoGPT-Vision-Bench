#!/bin/bash

# 1. 경로 설정 (HuatuoGPT-Vision-Bench 내부 기준)
PROJECT_ROOT=$(pwd)
REPO_NAME="HuatuoGPT-Vision"
REPO_PATH="$PROJECT_ROOT/$REPO_NAME"
DATA_DIR="$REPO_PATH/data"

echo "📂 작업 경로: $PROJECT_ROOT"
echo "📂 원본 저장 경로: $REPO_PATH"

# 2. 원본 레포지토리 클론 (없을 경우에만 실행)
if [ ! -d "$REPO_PATH" ]; then
    echo "🚀 [1/3] 원본 HuatuoGPT-Vision 레포지토리 클론 중..."
    git clone https://github.com/FreedomIntelligence/HuatuoGPT-Vision.git "$REPO_PATH"
else
    echo "✅ [1/3] 원본 레포지토리가 이미 존재합니다."
fi

# 3. 필수 도구 및 데이터 디렉토리 준비
apt-get update && apt-get install -y unzip wget
pip install datasets tqdm pillow setuptools

mkdir -p "$DATA_DIR/slake"
mkdir -p "$DATA_DIR/vqarad/imgs"

# 4. SLAKE 데이터셋 처리 (폴더 구조 유지하며 압축 해제)
echo "🚀 [2/3] SLAKE 데이터셋 수동 다운로드 및 압축 해제..."
SLAKE_DIR="$DATA_DIR/slake"

# 이미지 압축파일 및 JSON 직접 다운로드
wget -O "$SLAKE_DIR/imgs.zip" https://huggingface.co/datasets/BoKelvin/SLAKE/resolve/main/imgs.zip
wget -O "$SLAKE_DIR/raw_test.json" https://huggingface.co/datasets/BoKelvin/SLAKE/resolve/main/test.json

# [중요] -j 옵션을 제거하여 xmlab1/source.jpg 같은 폴더 구조를 유지하며 imgs 폴더에 풉니다.
unzip -o "$SLAKE_DIR/imgs.zip" -d "$SLAKE_DIR/imgs"
rm "$SLAKE_DIR/imgs.zip"

python3 -c "
import json, os
slake_dir = '$SLAKE_DIR'
with open(os.path.join(slake_dir, 'raw_test.json'), 'r', encoding='utf-8') as f:
    data = json.load(f)

# 영어 질문만 필터링하고 경로 필드 생성
formatted = []
for item in data:
    if item.get('q_lang') == 'en':
        item['image_name'] = item['img_name']  # 'xmlab1/source.jpg' 구조 그대로 유지
        formatted.append(item)

with open(os.path.join(slake_dir, 'test.json'), 'w', encoding='utf-8') as f:
    json.dump(formatted, f, indent=4)
print(f'✅ SLAKE 전처리 완료: {len(formatted)}개 영어 데이터 확보')
"

# 5. VQA-RAD 처리
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

echo "🎉 모든 준비 완료! 이제 scripts/ 안의 평가 코드를 돌리시면 됩니다."
