# 🏥 Medical-VQA-Bench

HuatuoGPT-Vision 모델을 활용한 의료 VQA (SLAKE, VQA-RAD) 성능 평가 파이프라인입니다. 
클라우드 GPU 인스턴스(e.g., Vast.ai)에서 즉시 실험이 가능하도록 자동화된 환경 및 데이터 구축 스크립트를 제공합니다.

## 📂 프로젝트 구조 (Project Structure)

```text
Medical-VQA-Bench/
├── setup/
│   ├── environment.yml       # Conda 환경 설정 (PyTorch 2.0.1 기반)
│   └── download_assets.sh    # 데이터셋 & 모델 가중치 자동 다운로드
├── scripts/
│   ├── eval_slake.py         # SLAKE 데이터셋 배치 평가 (Batch 4)
│   └── eval_vqarad.py        # VQA-RAD 데이터셋 배치 평가 (Batch 4)
├── results/                  # 평가 결과 CSV 자동 저장 폴더
└── README.md                 # 프로젝트 실행 가이드

'''
conda env create -f setup/environment.yml
conda activate huatuo
'''

'''
chmod +x setup/download_assets.sh
./setup/download_assets.sh
'''

'''
# SLAKE 데이터셋 평가 실행
python scripts/eval_slake.py

# VQA-RAD 데이터셋 평가 실행
python scripts/eval_vqarad.py
'''