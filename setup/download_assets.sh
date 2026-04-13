#!/bin/bash

# 1. 경로 설정 (프로젝트 루트 기준)
PROJECT_ROOT=$(pwd)
WORKSPACE_ROOT=$(dirname "$PROJECT_ROOT")
REPO_PATH="$WORKSPACE_ROOT/HuatuoGPT-Vision"
DATA_DIR="$REPO_PATH/data"

echo "📂 작업 경로: $PROJECT_ROOT"
echo "📂 데이터 저장 경로: $DATA_DIR"

# 2. 필수 라이브러리 설치
pip install datasets tqdm pillow

# 3. 데이터셋 통합 다운로드 및 변환 (Python 스크립트 실행)
echo "🚀 [1/2] SLAKE & VQA-RAD 데이터셋 다운로드 및 변환 시작..."

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
    
    # Hugging Face에서 데이터 로드
    ds = load_dataset(ds_name, split='test')
    formatted_data = []
    
    for i, item in enumerate(tqdm(ds)):
        # 이미지 저장
        img_filename = f'{save_name}_test_{i}.jpg'
        item['img'].convert('RGB').save(os.path.join(img_dir, img_filename))
        
        # 필드 구성 (SLAKE/VQA-RAD 공통 규격화)
        ans = str(item['answer']).strip()
        # answer_type이 없으면 yes/no 여부로 자동 생성
        ans_type = item.get('answer_type', 'CLOSED' if ans.lower() in ['yes', 'no'] else 'OPEN')
        
        formatted_data.append({
            'qid': f'{save_name}_{i}',
            'image_name': img_filename, # 우리 코드는 image_name으로 통일
            'question': item['question'],
            'answer': ans,
            'answer_type': ans_type
        })
    
    # JSON 저장
    with open(os.path.join(target_path, f'{save_name}_test.json'), 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, indent=4)
    print(f'✅ {ds_name} 완료!')

# 실행
process_dataset('BoKelvin/SLAKE', 'slake')
process_dataset('flaviagiammarino/vqa-rad', 'vqarad')
"

# 4. 모델 가중치 미리 로드 (캐싱)
echo "🚀 [2/2] HuatuoGPT-Vision 모델 가중치 체크..."
python3 -c "
from transformers import AutoTokenizer
model_id = 'FreedomIntelligence/HuatuoGPT-Vision-7b'
AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
"

echo "🎉 모든 데이터와 모델 준비가 완료되었습니다!"