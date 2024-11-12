# Yolov8_video

# 1. 필요한 라이브러리 설치
%pip install ultralytics opencv-python-headless matplotlib yt-dlp

import yt_dlp
import cv2
from google.colab import files
from ultralytics import YOLO
from IPython.display import Video, display

# 2. 유튜브 비디오 링크
video_url = "https://youtube.com/shorts/KnyMNQklJrc?si=w7UVTSEKV0fk-G2I"

# 3. 유튜브 비디오 다운로드 (yt-dlp 사용)
ydl_opts = {
    'format': 'best',  # 비디오 품질을 최대로 다운로드
    'outtmpl': '/content/sample_data/%(id)s.%(ext)s',  # 다운로드된 비디오 경로
}

# 4. 유튜브 비디오 다운로드
with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    info_dict = ydl.extract_info(video_url, download=True)
    video_filename = f"/content/sample_data/{info_dict['id']}.mp4"  # 다운로드된 비디오 파일 경로

# 5. YOLO 모델 로드 (물체 감지용)
model = YOLO('yolov8n.pt')  # yolov8n.pt 모델 파일 로드

# 6. 비디오 파일 열기
cap = cv2.VideoCapture(video_filename)

# 7. 비디오 출력 설정
output_video_path = '/content/output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # mp4 포맷
out = cv2.VideoWriter(output_video_path, fourcc, 30, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

# 8. YOLO를 이용한 객체 감지 (모든 객체 감지)
    results = model(frame)  # 이미지를 직접 전달하여 예측
    annotations = results[0].boxes  # 예측된 박스 정보

# 9. 각 객체에 대해 사각형을 그리고 레이블을 표시
    for annotation in annotations:
        x1, y1, x2, y2 = map(int, annotation.xyxy[0])  # 좌표
        label = f'{model.names[int(annotation.cls)]}'  # 클래스 이름
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # 감지된 물체를 빨간색 사각형으로 표시
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)  # 레이블 출력

# 10. 결과 비디오 파일에 프레임 저장
    out.write(frame)

# 11. 비디오 처리 종료
cap.release()
out.release()

# 12. 결과 비디오를 Colab에서 표시
display(Video(output_video_path))

# 13. 결과 파일 다운로드 링크 제공
files.download(output_video_path)


https://github.com/user-attachments/assets/cc69dcc2-ab4a-42fd-83f8-fdeea8b796d8

