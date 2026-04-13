#!/bin/bash

# 1. 경로 설정 (HuatuoGPT-Vision-Bench 내부 구조)
PROJECT_ROOT=$(pwd)
REPO_NAME="HuatuoGPT-Vision"
REPO_PATH="$PROJECT_ROOT/$REPO_NAME"

echo "📂 원본 저장 및 데이터 경로: $REPO_PATH"

# 2. 원본 모델 레포지토리가 없으면 클론 (Bench 폴더 내부로)
if [ ! -d "$REPO_PATH" ]; then
    echo "🚀 [1/2] 원본 레포지토리 클론 중..."
    git clone https://github.com/FreedomIntelligence/HuatuoGPT-Vision.git "$REPO_PATH"
else
    echo "✅ [1/2] 원본 레포지토리가 이미 존재합니다."
fi

# 3. 데이터셋 다운로드 및 필드 에러 해결 (Python)
echo "🚀 [2/2] 데이터셋 다운로드 및 필드 자동 분석 시작..."
DATA_DIR="$REPO_PATH/data"

python3 -c "
import os, json
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image

data_dir = '$DATA_DIR'

def process_dataset(ds_name, save_name):
    print(f'\n--- {ds_name} 처리 중 ---')
    target_path = os.path.join(data_dir, save_name)
    img_dir = os.path.join(target_path, 'imgs')
    os.makedirs(img_dir, exist_ok=True)
    
    # 데이터 로드
    ds = load_dataset(ds_name, split='test')
    
    # [핵심] 이미지 필드명 자동 탐색 (img 혹은 image)
    sample = ds[0]
    img_key = next((k for k in ['img', 'image', 'raw_image'] if k in sample), None)
    
    if not img_key:
        print(f'❌ 에러: {ds_name}에서 이미지 필드를 찾을 수 없습니다.')
        return

    formatted_data = []
    for i, item in enumerate(tqdm(ds)):
        # SLAKE 영어 데이터만 필터링
        if save_name == 'slake' and item.get('q_lang') == 'zh':
            continue

        img_filename = f'{save_name}_test_{i}.jpg'
        
        try:
            # 이미지 저장
            img_obj = item[img_key]
            img_obj.convert('RGB').save(os.path.join(img_dir, img_filename))
            
            # 공통 데이터 규격화
            formatted_data.append({
                'qid': f'{save_name}_{i}',
                'image_name': img_filename,
                'question': item['question'],
                'answer': str(item['answer']),
                'answer_type': item.get('answer_type', 'CLOSED' if str(item['answer']).lower() in ['yes', 'no'] else 'OPEN')
            })
        except Exception as e:
            continue
    
    # 평가 코드가 바로 인식하도록 'test.json'으로 저장
    with open(os.path.join(target_path, 'test.json'), 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, indent=4)
    print(f'✅ {save_name} 완료!')

process_dataset('BoKelvin/SLAKE', 'slake')
process_dataset('flaviagiammarino/vqa-rad', 'vqarad')
"
echo "🎉 모든 데이터 준비 완료! 이제 scripts/ 평가 코드를 돌리세요."