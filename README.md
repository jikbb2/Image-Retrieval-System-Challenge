# Image-Retrieval-System-Challenge
An advanced image classification and retrieval system built with a hybrid ensemble of Early Vision and Deep Learning models, optimized with object detection, multi-stage dimensionality reduction, and cross-validation.
# 고성능 이미지 분류 및 검색 시스템 (Advanced Image Classification & Retrieval System)

## 1. 프로젝트 개요 (Overview)

본 프로젝트는 복잡한 배경 노이즈를 포함하는 **reCAPTCHA 이미지 데이터셋**을 대상으로, 가장 높은 정확도와 일반화 성능을 가진 이미지 분류 및 검색 시스템을 구축하는 것을 목표로 합니다. (데이터셋 출처: [GitHub - folfcoder/recaptcha-dataset](https://github.com/folfcoder/recaptcha-dataset))

단순한 피처 추출을 넘어, **Early Vision 피처**와 **딥러닝(ResNet18) 피처**의 장점을 결합한 **하이브리드 앙상블 모델**을 최종적으로 설계했습니다. 시스템의 성능을 극대화하기 위해 아래와 같은 핵심 기술들을 체계적으로 적용했습니다.

* **오브젝트 디텍션(YOLOv8)** 기반의 전처리로 배경 노이즈 제거
* **다단계 차원 축소** (RandomForest Feature Selection + PCA)
* **교차 검증**을 통한 하이퍼파라미터 최적화
* **Soft Voting 앙상블**을 통한 예측 정확도 향상

## 2. 핵심 기술 및 아키텍처 (Core Technologies & Architecture)

본 프로젝트는 두 가지 주요 챌린지를 해결하며 점진적으로 성능을 발전시킨 두 개의 핵심 파이프라인으로 구성되어 있습니다.

### 챌린지 1: Early Vision 하이브리드 앙상블 파이프라인

* **하이브리드 피처 추출**: SIFT/AKAZE(키포인트), HOG(형태), GLCM/LBP/Law's(질감), Color Histogram(색상) 등 다차원적인 정보를 동시에 추출하여 이미지의 복합적인 특징을 표현합니다.
* **피처별 최적화**: 각 피처의 성능을 극대화하기 위해 CLAHE 등 맞춤형 전처리 기법을 적용했습니다.
* **모델 최적화**: 교차 검증을 통해 각 피처 그룹별 최적의 PCA 차원을 탐색하고, Soft Voting 앙상블로 최종 예측을 수행하여 정확도를 극대화했습니다.

### 챌린지 2: 딥러닝(Fine-tuning) 기반 파이프라인

* **파인튜닝된 특징 추출기**: 사전 학습된 ResNet18 모델을 reCAPTCHA 데이터셋에 맞게 파인튜닝하여, 데이터의 특정 도메인에 고도로 전문화된 피처를 추출합니다.
* **성능 최적화**: 추출된 고차원 딥러닝 피처에 대해 표준화(Standardization) 및 교차 검증 기반 PCA를 적용하여 k-NN 매칭에 가장 효율적인 형태로 압축합니다.

## 3. 실행 가이드 (How to Run)

본 프로젝트는 Google Colab 환경에서 개발되었습니다. 각 챌린지의 코드는 별도의 Jupyter Notebook (`ImageMatchingSystem_EarlyVision.ipynb`, `ImageMatchingSystem_DL.ipynb`)으로 구성되어 있습니다.

### Step 1: 환경 설정

각 노트북 상단에 있는 환경 설정 셀을 실행하여 필요한 라이브러리를 설치하고 데이터셋을 다운로드합니다.

### Step 2: 모델 학습 및 DB 구축 (챌린지 전)

챌린지를 수행하기 전에, 각 노트북의 학습 관련 함수를 실행하여 모델과 피처 데이터베이스를 미리 구축해야 합니다.

* **챌린지 1 모델 학습 (`Early Vision`)**
    * `ImageMatchingSystem_EarlyVision.ipynb` 노트북에서 아래 함수를 실행합니다.
    ```python
    # Early Vision 피처 DB 및 중요도 모델 생성
    build_feature_database()
    ```

* **챌린지 2 모델 학습 (`Deep Learning`)**
    * `ImageMatchingSystem_DL.ipynb` 노트북에서 아래 함수들을 순서대로 실행합니다.
    ```python
    # 1. 데이터 로드 및 분할
    dataloaders, train_data, test_data, target_labels = prepare_dataloaders_and_split_data()

    # 2. 모델 파인튜닝
    finetuned_model_path = fine_tune_and_save_model(dataloaders, target_labels)

    # 3. 피처 DB 구축 및 최적화
    build_and_optimize_feature_db(finetuned_model_path, train_data)
    ```

### Step 3: 최종 평가

새로운 테스트 데이터셋 폴더가 주어지면, 각 노트북의 평가 함수를 실행하여 결과 CSV 파일을 생성합니다.

* **챌린지 1 모델로 평가**
    * `ImageMatchingSystem_EarlyVision.ipynb` 노트북에서 아래 함수를 실행합니다.
    ```python
    # 저장된 Early Vision 모델로 새로운 테스트셋 평가
    evaluate_final_model(test_dataset_path='./path/to/test_images')
    ```

* **챌린지 2 모델로 평가**
    * `ImageMatchingSystem_DL.ipynb` 노트북에서 아래 함수를 실행합니다.
    ```python
    # 저장된 Deep Learning 모델로 새로운 테스트셋 평가
    evaluate_on_test_data(test_data, target_labels)
    # (주의: 이 함수는 Step 2에서 반환된 test_data 변수가 필요합니다)
    ```

## 4. 요구사항 (Requirements)

실행에 필요한 라이브러리 목록입니다. (`requirements.txt`)

```
torch
torchvision
ultralytics
opencv-python
opencv-contrib-python
scikit-learn
scikit-image
numpy
pandas
joblib
tqdm
matplotlib
