import cv2
import numpy as np
import sys

# 사각형(번호판) 모양 검출 함수
def shape_detect(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.03 * peri, True)  # 윤곽선의 단순화
    shape = "undefined"
    
    # 윤곽선이 사각형(꼭짓점 4개에서 8개 사이)인지 확인
    if 4 <= len(approx) <= 8:
        (x, y, w, h) = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        
        # 사각형의 가로세로 비율이 번호판의 비율과 유사하면 'rectangle'로 설정
        if 4 < aspect_ratio < 6:  # 비율을 4:1에서 6:1 사이로 설정
            shape = "rectangle"
    
    return shape

# 이미지 전처리 (그레이스케일 변환 및 캐니 엣지 검출)
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    return edged

# 번호판 검출 함수
def detect_license_plate(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"이미지를 불러올 수 없습니다: {image_path}")
        return
    
    # 이미지 전처리
    edged = preprocess_image(img)
    
    # 윤곽선 검출
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        shape = shape_detect(contour)
        
        if shape == "rectangle":
            # 윤곽선의 외곽 사각형 좌표 얻기
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cx, cy = x + w // 2, y + h // 2
            cv2.putText(img, "License", (cx - 50, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # 결과 출력
    cv2.imshow(f'License Plate Detection - {image_path}', img)
    cv2.imshow(f'Edge Detection - {image_path}', edged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 이미지 경로 리스트
image_paths = [
    '00.jpg',
    '01.jpg',
    '02.jpg',
    '03.jpg',
    '04.jpg',
    '05.jpg'
]

# 각 이미지에 대해 번호판 검출 수행
for image_path in image_paths:
    detect_license_plate(image_path)
