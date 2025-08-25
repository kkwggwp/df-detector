# app.py (웹 서버 파드용)

import os
from flask import Flask, request, jsonify, render_template, url_for
from werkzeug.utils import secure_filename
import requests  # GPU 파드 호출을 위해 requests 라이브러리 추가

app = Flask(__name__)

# --- 설정 (모델 로딩 부분 없음) ---
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# 💥 중요: GPU 파드의 내부 쿠버네티스 서비스 주소
# 'gpu-service'는 쿠버네티스에서 생성할 Service의 이름입니다.
GPU_SERVICE_URL = "http://gpu-service/predict"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# 파일 자동 삭제 스케줄러는 그대로 유지해도 좋습니다.
# ... (이전 코드와 동일) ...


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "파일이 없습니다."}), 400

    file = request.files["image"]
    model_name = request.form.get("model", "DINO")

    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "유효하지 않은 파일입니다."}), 400

    try:
        # --- ▼ 여기가 핵심 변경점 ▼ ---
        # 파일을 직접 처리하는 대신, GPU 파드로 그대로 전달(proxy)합니다.
        files = {"image": (file.filename, file.read(), file.content_type)}
        payload = {"model": model_name}

        # GPU 서비스에 POST 요청을 보냅니다.
        response = requests.post(GPU_SERVICE_URL, files=files, data=payload, timeout=30)
        response.raise_for_status()  # 2xx 응답이 아니면 에러 발생

        result = response.json()
        # --- ▲ 여기가 핵심 변경점 ▲ ---

        # 임시 파일 저장은 이제 필요 없지만, 사용자에게 이미지를 다시 보여주려면 필요합니다.
        # 이 부분은 로직에 따라 선택적으로 구현할 수 있습니다.

        return jsonify(result)

    except requests.exceptions.RequestException as e:
        # GPU 파드와 통신 실패 시 오류 처리
        return jsonify({"error": f"모델 서버와 통신 중 오류 발생: {e}"}), 502
    except Exception as e:
        return jsonify({"error": f"알 수 없는 오류 발생: {str(e)}"}), 500


if __name__ == "__main__":
    # 💥 중요: 모델 로딩 함수 호출 제거!
    # load_models()
    app.run(host="0.0.0.0", port=5000)  # 외부 접근을 위해 host 변경
