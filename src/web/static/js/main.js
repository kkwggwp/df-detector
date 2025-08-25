document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('upload-form');
    const imageUpload = document.getElementById('image-upload');
    const modelSelect = document.getElementById('model-select');
    const modelDescription = document.getElementById('model-description');
    
    const loadingSection = document.getElementById('loading');
    const resultSection = document.getElementById('result-section');
    
    const uploadedImage = document.getElementById('uploaded-image');
    const resultText = document.getElementById('result-text');
    const confidenceFill = document.getElementById('confidence-fill');
    const confidenceScore = document.getElementById('confidence-score');

    const modelDescriptions = {
        'ViT': '<strong>ViT:</strong> 이미지를 여러 조각(패치)으로 나누어 전체적인 맥락과 관계를 분석하는 데 강점이 있는 모델입니다.',
        'DINO': '<strong>DINO:</strong> 자기지도학습(Self-supervised learning)을 통해 레이블 없는 데이터에서도 이미지의 핵심 특징을 학습하는 모델입니다.',
        'CLIP': '<strong>CLIP:</strong> 이미지와 텍스트를 함께 학습하여, 두 데이터 사이의 관계를 이해하고 다양한 시각적 개념을 인식하는 능력이 뛰어난 모델입니다.'
    };

    // 모델 선택 시 설명 변경
    modelSelect.addEventListener('change', () => {
        modelDescription.innerHTML = `<p>${modelDescriptions[modelSelect.value]}</p>`;
    });

    // 폼 제출 이벤트 처리
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault(); // 기본 폼 제출 방지

        const file = imageUpload.files[0];
        if (!file) {
            alert('이미지 파일을 선택해주세요.');
            return;
        }

        // 로딩 화면 표시, 이전 결과 숨기기
        loadingSection.classList.remove('hidden');
        resultSection.classList.add('hidden');

        const formData = new FormData();
        formData.append('image', file);
        formData.append('model', modelSelect.value);

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || '서버 오류가 발생했습니다.');
            }

            const data = await response.json();
            displayResults(data, file);

        } catch (error) {
            alert(`오류 발생: ${error.message}`);
        } finally {
            // 로딩 화면 숨기기
            loadingSection.classList.add('hidden');
        }
    });

    function displayResults(data, file) {
        // 이미지 미리보기 설정
        const reader = new FileReader();
        reader.onload = (e) => {
            uploadedImage.src = e.target.result;
        };
        reader.readAsDataURL(file);

        // 결과 텍스트 및 신뢰도 표시
        const percentage = (data.confidence * 100).toFixed(2);
        resultText.textContent = `결과: ${data.label}`;
        confidenceScore.textContent = `신뢰도: ${percentage}%`;
        
        // 신뢰도 바 업데이트
        confidenceFill.style.width = `${percentage}%`;
        
        // 결과에 따라 색상 변경
        confidenceFill.classList.remove('real', 'fake', 'ambiguous');
        if (data.label === '실제') {
            confidenceFill.classList.add('real');
        } else if (data.label === '딥페이크') {
            confidenceFill.classList.add('fake');
        } else {
            confidenceFill.classList.add('ambiguous');
        }
        
        // 결과 섹션 표시
        resultSection.classList.remove('hidden');
    }
});