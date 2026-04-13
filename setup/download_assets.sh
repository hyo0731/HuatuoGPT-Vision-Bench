#!/bin/bash

# 1. 경로 설정 (HuatuoGPT-Vision-Bench 내부 구조)
PROJECT_ROOT=$(pwd)
REPO_NAME="HuatuoGPT-Vision"
REPO_PATH="$PROJECT_ROOT/$REPO_NAME"

echo "📂 원본 저장 및 데이터 경로: $REPO_PATH"

# 2. 원본 모델 레포지토리 클론
if [ ! -d "$REPO_PATH" ]; then
    echo "🚀 [1/2] 원본 레포지토리 클론 중..."
    git clone https://github.com/FreedomIntelligence/HuatuoGPT-Vision.git "$REPO_PATH"
else
    echo "✅ [1/2] 원본 레포지토리가 이미 존재합니다."
fi

# 3. 데이터셋 다운로드 (이미지 필드 강제 탐색 로직 추가)
echo "🚀 [2/2] 데이터셋 다운로드 및 필드 정밀 분석 시작..."
DATA_DIR="$REPO_PATH/data"

python3 -c "
import os, json
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image

data_dir = '$DATA_DIR'

def process_dataset(ds_name, save_name):
    print(f'\n--- {ds_name} 분석 중 ---')
    target_path = os.path.join(data_dir, save_name)
    img_dir = os.path.join(target_path, 'imgs')
    os.makedirs(img_dir, exist_ok=True)
    
    # 1. 데이터 로드
    ds = load_dataset(ds_name, split='test')
    
    # 2. 첫 번째 아이템의 모든 키 출력 (디버깅용)
    sample = ds[0]
    print(f'🔍 발견된 필드 목록: {list(sample.keys())}')
    
    # 3. 이미지 필드 찾기 (우선순위: image -> img -> PIL객체 타입)
    img_key = None
    for k in ['image', 'img', 'raw_image']:
        if k in sample:
            img_key = k
            break
            
    if not img_key:
        # 키 이름으로 못 찾으면 데이터 타입을 직접 확인
        for k, v in sample.items():
            if isinstance(v, Image.Image):
                img_key = k
                break

    if not img_key:
        print(f'❌ 에러: {ds_name}에서 이미지 데이터를 찾을 수 없습니다.')
        return

    print(f'✅ 사용될 이미지 필드명: \"{img_key}\"')

    formatted_data = []
    for i, item in enumerate(tqdm(ds)):
        # SLAKE 영어 데이터만 추출
        if save_name == 'slake' and item.get('q_lang') != 'en':
            continue

        img_filename = f'{save_name}_test_{i}.jpg'
        
        try:
            # 이미지 저장
            img_obj = item[img_key]
            if not isinstance(img_obj, Image.Image):
                 continue
            
            img_obj.convert('RGB').save(os.path.join(img_dir, img_filename))
            
            formatted_data.append({
                'qid': f'{save_name}_{i}',
                'image_name': img_filename,
                'question': item['question'],
                'answer': str(item['answer']),
                'answer_type': item.get('answer_type', 'CLOSED' if str(item['answer']).lower() in ['yes', 'no'] else 'OPEN')
            })
        except Exception as e:
            continue
    
    with open(os.path.join(target_path, 'test.json'), 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, indent=4)
    print(f'✅ {save_name} 완료! (총 {len(formatted_data)}개 추출)')

process_dataset('BoKelvin/SLAKE', 'slake')
process_dataset('flaviagiammarino/vqa-rad', 'vqarad')
"
echo "🎉 준비 완료!"