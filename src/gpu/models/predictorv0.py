import torch
import torchvision.transforms as transforms
from PIL import Image
import random
import time
import os

# --------------------------------------------------------------------------
# ★★ 중요 ★★
# 이 파일은 실제 모델을 로드하고 추론을 수행하는 핵심 로직을 담당합니다.
# 1. DINO 모델은 실제 파일(`DINO_v1.pth`)을 로드하여 추론합니다.
# 2. ViT, CLIP은 나중에 모델 파일을 추가하고 TODO 부분을 채우면 바로 작동합니다.
# --------------------------------------------------------------------------

# 모델 파일들이 저장된 경로
MODEL_PATH_DIR = 'model_files'
MODEL_PATHS = {
    "DINO": os.path.join(MODEL_PATH_DIR, "DINO_v1.pth"),
    "ViT": os.path.join(MODEL_PATH_DIR, "ViT_v1.pth"),
    "CLIP": os.path.join(MODEL_PATH_DIR, "CLIP_v1.pth")
}

# 로드된 모델들을 저장할 딕셔너리
MODELS = {}

# 이미지 전처리 (DINO/ViT 모델은 일반적으로 이런 전처리 과정을 거칩니다)
# 팀에서 사용한 전처리 방식에 맞게 수정이 필요할 수 있습니다.
TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_models():
    """서버 시작 시 모델들을 미리 로드하여 메모리에 올려놓는 함수"""
    print("Loading models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- 1. DINO 모델 로드 ---
    dino_path = MODEL_PATHS["DINO"]
    if os.path.exists(dino_path):
        try:
            # dino_vitb16 모델 구조를 불러옵니다. 팀에서 사용한 아키텍처로 변경해야 할 수 있습니다.
            # 예: dino_vits16, dino_vitb8 등
            model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
            
            # 딥페이크 판별을 위해 추가한 classification head의 out_features를 맞춰야 합니다.
            # 예시: 2개의 클래스(real/fake)로 판별하는 경우
            model.head = torch.nn.Linear(model.head.in_features, 2)
            
            # 학습된 가중치(state_dict)를 불러옵니다.
            model.load_state_dict(torch.load(dino_path, map_location=device))
            model.to(device)
            model.eval() # 추론 모드로 설정
            MODELS["DINO"] = {"model": model, "device": device}
            print(f"✅ DINO model loaded successfully from {dino_path}")
        except Exception as e:
            print(f"❗️ Error loading DINO model: {e}")
            MODELS["DINO"] = None
    else:
        print(f"❗️ DINO model file not found at {dino_path}")
        MODELS["DINO"] = None
        
    # --- 2. ViT 모델 로드 (현재는 Placeholder) ---
    # TODO: ViT_v1.pth 파일이 준비되면 아래 주석을 풀고 실제 모델 로드 코드를 작성하세요.
    MODELS["ViT"] = None
    print("ViT model is currently a placeholder.")
    # if os.path.exists(MODEL_PATHS["ViT"]):
    #     try:
    #         # 여기에 ViT 모델 구조를 정의하고 가중치를 로드하는 코드 작성
    #         MODELS["ViT"] = ...
    #         print("✅ ViT model loaded successfully.")
    #     except Exception as e:
    #         print(f"❗️ Error loading ViT model: {e}")

    # --- 3. CLIP 모델 로드 (현재는 Placeholder) ---
    # TODO: CLIP_v1.pth 파일이 준비되면 아래 주석을 풀고 실제 모델 로드 코드를 작성하세요.
    MODELS["CLIP"] = None
    print("CLIP model is currently a placeholder.")


def run_prediction(image_path, model_name):
    """
    이미지 경로와 모델 이름을 받아 딥페이크 여부를 판별합니다.
    """
    print(f"Running prediction with model: {model_name} on image: {image_path}")

    # 선택된 모델이 로드되었는지 확인
    if model_name not in MODELS or MODELS[model_name] is None:
        # 모델이 로드되지 않았다면, 임시 결과 반환 (가짜 추론)
        print(f"Warning: {model_name} model not loaded. Returning mock result.")
        time.sleep(random.uniform(0.5, 1.5))
        confidence = random.random()
        label = '딥페이크' if confidence > 0.7 else ('실제' if confidence < 0.3 else '애매')
        return {'label': label, 'confidence': confidence}

    # --- 실제 모델 추론 실행 ---
    try:
        # 1. 이미지 로드 및 전처리
        image = Image.open(image_path).convert('RGB')
        image_tensor = TRANSFORMS(image).unsqueeze(0) # 배치 차원 추가

        # 2. DINO 모델 추론 로직
        if model_name == "DINO":
            model_info = MODELS["DINO"]
            model, device = model_info["model"], model_info["device"]
            image_tensor = image_tensor.to(device)

            with torch.no_grad():
                output = model(image_tensor)
                # Softmax를 통해 확률값으로 변환
                probabilities = torch.nn.functional.softmax(output, dim=1)
            
            # 딥페이크일 확률을 confidence로 사용 (클래스 1을 딥페이크로 가정)
            # 클래스 순서(0:real, 1:fake)는 팀의 학습 방식에 따라 달라질 수 있습니다.
            confidence = probabilities[0][1].item() 

            if confidence > 0.7:
                label = '딥페이크'
            elif confidence < 0.3:
                label = '실제'
            else:
                label = '애매'
        
        # TODO: ViT 모델 추론 로직 추가
        elif model_name == "ViT":
            # 여기에 ViT 모델을 사용한 추론 코드 작성
            raise NotImplementedError("ViT model prediction logic is not implemented yet.")

        # TODO: CLIP 모델 추론 로직 추가
        elif model_name == "CLIP":
            # 여기에 CLIP 모델을 사용한 추론 코드 작성
            raise NotImplementedError("CLIP model prediction logic is not implemented yet.")
        
        else:
             return {'error': '알 수 없는 모델입니다.'}

        print(f"Prediction result: {label}, Confidence: {confidence:.2f}")
        return {'label': label, 'confidence': confidence}

    except Exception as e:
        print(f"Error during prediction: {e}")
        return {'error': f'추론 중 오류 발생: {e}'}