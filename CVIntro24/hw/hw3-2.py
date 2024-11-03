import cv2
import numpy as np
import sys

# 번호판 후보 영역 검출 함수
def verify_aspect_size(size):
    w, h = size
    if h == 0 or w == 0: 
        return False
    
    aspect = h / w if h > w else w / h  # 종횡비 계산
    area = w * h

    # 번호판의 면적 및 가로 세로 비율 조건 설정
    area_condition = 3000 < area < 12000
    aspect_condition = 2.0 < aspect < 6.5  # 번호판 비율에 맞는 조건

    return area_condition and aspect_condition

# 사용자로부터 자동차 이미지 번호 입력받기
car_no = str(input("자동차 영상 번호 (00~05): "))

# 이미지 경로 설정 및 이미지 로드
img = cv2.imread('cars/' + car_no + '.jpg')
if img is None:
    sys.exit(f"Error: Could not load image with car number {car_no}.")

# 이미지 전처리: 그레이스케일 변환, 가우시안 블러 적용, Canny 에지 검출
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blur, 100, 200)

# 윤곽선 찾기
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 번호판 후보 필터링 및 최소면적 사각형 찾기
candidates = []
for contour in contours:
    # 최소면적 사각형 구하기
    rect = cv2.minAreaRect(contour)  # (중심, (가로, 세로), 회전각도) 반환
    box = cv2.boxPoints(rect)        # 사각형의 4개의 꼭짓점 좌표
    box = np.int32(box)               # 정수로 변환

    # 가로와 세로 정보 추출
    w, h = rect[1]
    if verify_aspect_size((w, h)):   # 조건 만족 여부 확인
        candidates.append(box)       # 후보 영역 추가

# 후보 영역 그리기
for candidate in candidates:
    cv2.drawContours(img, [candidate], 0, (0, 255, 0), 2)  # 번호판 후보 영역 표시

# 결과 이미지 출력
cv2.imshow("License Plate Candidates", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
