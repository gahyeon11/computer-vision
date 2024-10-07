import numpy as np
import cv2
import sys

# 600x900 크기의 흰색 이미지 생성
img = np.ones((600, 900, 3), np.uint8) * 255

# 전역 변수들
BrushSiz = 5  # 붓의 크기
LColor, RColor = (255, 0, 0), (0, 0, 255)  # 파란색과 빨간색
GColor, YColor = (0, 255, 0), (0, 255, 255)  # 초록색과 노란색
ix, iy = -1, -1  

# 두 점 사이의 거리 계산 함수
def calculate_distance(x1, y1, x2, y2):
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def painting(event, x, y, flags, param):
    global ix, iy, img  

    # Alt + 마우스 왼쪽 클릭: 직사각형 그리기
    if event == cv2.EVENT_LBUTTONDOWN and flags & cv2.EVENT_FLAG_ALTKEY:
        ix, iy = x, y  
    elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_ALTKEY and flags & cv2.EVENT_FLAG_LBUTTON:
        temp_img = img.copy()
        cv2.rectangle(temp_img, (ix, iy), (x, y), (0, 0, 0), 2)
        cv2.imshow('Painting', temp_img)
    elif event == cv2.EVENT_LBUTTONUP and flags & cv2.EVENT_FLAG_ALTKEY:
        cv2.rectangle(img, (ix, iy), (x, y), (0, 0, 0), 2)

    # Alt + 마우스 오른쪽 클릭: 채워진 직사각형 그리기
    elif event == cv2.EVENT_RBUTTONDOWN and flags & cv2.EVENT_FLAG_ALTKEY:
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_ALTKEY and flags & cv2.EVENT_FLAG_RBUTTON:
        temp_img = img.copy()
        cv2.rectangle(temp_img, (ix, iy), (x, y), (0, 0, 0), -1)
        cv2.imshow('Painting', temp_img)
    elif event == cv2.EVENT_RBUTTONUP and flags & cv2.EVENT_FLAG_ALTKEY:
        cv2.rectangle(img, (ix, iy), (x, y), (0, 0, 0), -1)

    # Ctrl + 마우스 왼쪽 클릭: 원 그리기
    elif event == cv2.EVENT_LBUTTONDOWN and flags & cv2.EVENT_FLAG_CTRLKEY:
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_CTRLKEY and flags & cv2.EVENT_FLAG_LBUTTON:
        temp_img = img.copy()
        radius = int(calculate_distance(ix, iy, x, y))
        cv2.circle(temp_img, (ix, iy), radius, (0, 0, 0), 2)
        cv2.imshow('Painting', temp_img)
    elif event == cv2.EVENT_LBUTTONUP and flags & cv2.EVENT_FLAG_CTRLKEY:
        radius = int(calculate_distance(ix, iy, x, y))
        cv2.circle(img, (ix, iy), radius, (0, 0, 0), 2)

    # Ctrl + 마우스 오른쪽 클릭: 채워진 원 그리기
    elif event == cv2.EVENT_RBUTTONDOWN and flags & cv2.EVENT_FLAG_CTRLKEY:
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_CTRLKEY and flags & cv2.EVENT_FLAG_RBUTTON:
        temp_img = img.copy()
        radius = int(calculate_distance(ix, iy, x, y))
        cv2.circle(temp_img, (ix, iy), radius, (0, 0, 0), -1)
        cv2.imshow('Painting', temp_img)
    elif event == cv2.EVENT_RBUTTONUP and flags & cv2.EVENT_FLAG_CTRLKEY:
        radius = int(calculate_distance(ix, iy, x, y))
        cv2.circle(img, (ix, iy), radius, (0, 0, 0), -1)

    # 마우스 왼쪽 클릭 + 이동: 파란색 원 그리기
    elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON and not (flags & cv2.EVENT_FLAG_SHIFTKEY or flags & cv2.EVENT_FLAG_ALTKEY or flags & cv2.EVENT_FLAG_CTRLKEY):
        cv2.circle(img, (x, y), BrushSiz, LColor, -1)

    # 마우스 오른쪽 클릭 + 이동: 빨간색 원 그리기
    elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_RBUTTON and not (flags & cv2.EVENT_FLAG_SHIFTKEY or flags & cv2.EVENT_FLAG_ALTKEY or flags & cv2.EVENT_FLAG_CTRLKEY):
        cv2.circle(img, (x, y), BrushSiz, RColor, -1)

    # Shift + 마우스 왼쪽 클릭 + 이동: 초록색 원 그리기
    elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON and flags & cv2.EVENT_FLAG_SHIFTKEY:
        cv2.circle(img, (x, y), BrushSiz, GColor, -1)

    # Shift + 마우스 오른쪽 클릭 + 이동: 노란색 원 그리기
    elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_RBUTTON and flags & cv2.EVENT_FLAG_SHIFTKEY:
        cv2.circle(img, (x, y), BrushSiz, YColor, -1)

    cv2.imshow('Painting', img)

cv2.namedWindow('Painting')
cv2.imshow('Painting', img)
cv2.setMouseCallback('Painting', painting)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):  # 's' 키로 이미지 저장
        cv2.imwrite('painting.png', img)
        print('이미지 저장 완료')
    elif key == ord('q'):  # 'q' 키로 프로그램 종료
        break

cv2.destroyAllWindows()
