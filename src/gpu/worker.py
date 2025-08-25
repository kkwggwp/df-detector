# worker.py (GPU 연산용 파드용)

import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# 기존의 모델 로딩 및 추론 로직을 그대로 사용합니다.
from models.predictor import load_models, run_prediction

app = Flask(__name__)

# 업로드된 파일을 임시로 저장할 공간 (파드 내)
UPLOAD_FOLDER = "worker_uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    if "image" not in request.files:
        return jsonify({"error": "파일이 없습니다."}), 400

    file = request.files["image"]
    model_name = request.form.get("model", "DINO")

    if file.filename == "":
        return jsonify({"error": "파일이 선택되지 않았습니다."}), 400

    try:
        # 안전한 파일 이름으로 파드 내부에 임시 저장
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        # 저장된 파일 경로를 이용해 추론 실행
        result = run_prediction(filepath, model_name)

        # 임시 파일 삭제
        os.remove(filepath)

        if "error" in result:
            return jsonify(result), 400

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"추론 중 오류 발생: {str(e)}"}), 500


if __name__ == "__main__":
    # 💥 중요: 서버 시작 시 모델들을 미리 로드합니다.
    print("Loading models for GPU worker...")
    load_models()
    print("Models loaded. Worker is ready.")
    app.run(host="0.0.0.0", port=8080)  # 웹서버와 다른 포트 사용
