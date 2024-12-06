import cv2
import numpy as np
import easyocr
import matplotlib.pyplot as plt

# EasyOCR Reader 초기화 (한국어와 영어 OCR 지원)
reader = easyocr.Reader(['ko', 'en'])

# 이미지를 matplotlib으로 표시하는 함수
def display_image(image, title="Image"):
    """
    이미지 표시 함수
    Args:
        image: 출력할 이미지 배열
        title: 이미지 창 제목
    """
    if len(image.shape) == 3:  # 컬러 이미지일 경우 RGB 변환
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
    plt.title(title)
    plt.axis("off")
    plt.show()

# 사각형 좌표 정렬 함수
def order_points(pts):
    """
    사각형 꼭짓점 좌표를 정렬
    Args:
        pts: 사각형의 네 꼭짓점 좌표 배열
    Returns:
        rect: 정렬된 좌표 배열 (좌상단, 우상단, 우하단, 좌하단 순)
    """
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # 좌상단
    rect[2] = pts[np.argmax(s)]  # 우하단
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # 우상단
    rect[3] = pts[np.argmax(diff)]  # 좌하단
    return rect

# Perspective 변환
def warp_perspective(img, points, scale=1.2):
    """
    번호판 영역을 똑바로 변환
    Args:
        img: 원본 이미지
        points: 번호판 영역의 꼭짓점 좌표
        scale: 확대 비율 (기본값: 1.2)
    Returns:
        warped: 변환된 번호판 이미지
    """
    rect = order_points(points)
    center = np.mean(rect, axis=0)
    scaled_rect = (rect - center) * scale + center

    scaled_rect = np.clip(scaled_rect, 0, [img.shape[1] - 1, img.shape[0] - 1])  # 경계값 제한
    scaled_rect = scaled_rect.astype(np.float32)

    (tl, tr, br, bl) = scaled_rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)

    maxWidth = int(max(widthA, widthB))
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    matrix = cv2.getPerspectiveTransform(scaled_rect, dst)  # 변환 행렬 생성
    warped = cv2.warpPerspective(img, matrix, (maxWidth, maxHeight))  # 변환 수행
    return warped

# 번호판 이미지 전처리
def preprocess_plate(plate_img):
    """
    번호판 이미지 전처리 (히스토그램 평활화 및 노이즈 제거)
    Args:
        plate_img: 번호판 이미지
    Returns:
        binary: 전처리된 이진화 이미지
    """
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)  # 히스토그램 평활화
    resized = cv2.resize(equalized, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)  # 확대
    _, binary = cv2.threshold(resized, 150, 255, cv2.THRESH_BINARY_INV)  # 이진화

    # 노이즈 제거 (열림 연산)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))  # 커널 크기 (4x4)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small)

    return binary

# OCR로 텍스트 읽기
def recognize_text(binary_img):
    """
    OCR을 이용해 텍스트 추출
    Args:
        binary_img: 전처리된 번호판 이미지
    Returns:
        결과 텍스트 (문자열)
    """
    inverted_img = cv2.bitwise_not(binary_img)  # OCR에 적합하도록 반전
    results = reader.readtext(inverted_img, detail=0)  # EasyOCR로 텍스트 인식
    return ' '.join(results) if results else ""

# 번호판 후보 영역 탐지
def find_plate_candidates(img):
    """
    번호판 후보 영역 탐지
    Args:
        img: 원본 이미지
    Returns:
        candidates: 후보 영역의 꼭짓점 좌표 리스트
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # 블러링으로 노이즈 제거
    edges = cv2.Canny(blurred, 100, 200)  # Canny 엣지 탐지

    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    height, width = img.shape[:2]

    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        w, h = rect[1]
        if w == 0 or h == 0:
            continue

        aspect_ratio = max(w / h, h / w)  # 너비-높이 비율
        area = cv2.contourArea(box)
        y_center = rect[0][1]  # 중심의 y 좌표

        # 번호판 조건 필터링
        if 1000 < area < 50000 and 2.0 < aspect_ratio < 6.5 and height * 0.3 < y_center < height * 0.8:
            candidates.append(box)

    display_image(edges, "Edges")  # 엣지 이미지 표시
    return candidates

# 번호판 탐지 메인 함수
def detect_license_plate(image_path):
    """
    번호판 탐지 및 텍스트 인식
    Args:
        image_path: 입력 이미지 경로
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot load image: {image_path}")
        return

    candidates = find_plate_candidates(img)

    if candidates:
        for candidate in candidates:
            warped = warp_perspective(img, candidate, scale=1.2)  # 번호판 정렬 및 확대
            binary = preprocess_plate(warped)  # 번호판 전처리
            text = recognize_text(binary)  # 번호판 텍스트 추출

            if len(text) >= 4:  # 최소 길이 조건
                print(f"Recognized Plate: {text}")
                display_image(binary, "Image")
                return
        print("No plate recognized.")  # 유효 번호판 없음
    else:
        print("No plate found.")  # 후보 영역 없음
        
    display_image(img, "Original Image")  # 원본 이미지 표시

# 실행
car_no = input("자동차 영상 번호 (00~09): ")
image_path = f"./cars/{car_no}.jpg"
detect_license_plate(image_path)
