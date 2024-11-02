import cv2
import numpy as np
import sys

# 비디오 파일 로드
cap = cv2.VideoCapture('rock.mp4')  # 손이 포함된 동영상 파일명으로 변경하세요
if not cap.isOpened():
    sys.exit("동영상을 찾을 수 없습니다.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 그레이스케일 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 가우시안 블러 적용 (노이즈 감소)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny 에지 검출
    canny = cv2.Canny(blurred, 50, 150)
    
    # 윤곽선 검출
    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 빈 마스크 생성
    mask = np.zeros_like(gray)

    # 손가락 개수를 기반으로 가위, 바위, 보 판별
    result_text = "Unknown"
    
    if contours:
        # 가장 큰 윤곽선을 손으로 간주
        hand_contour = max(contours, key=cv2.contourArea)

        # 손 윤곽선 내부를 흰색으로 채우기
        cv2.drawContours(mask, [hand_contour], -1, 255, thickness=cv2.FILLED)

        # 외곽선을 원본 프레임에 겹쳐서 표시
        frame_with_contour = frame.copy()
        cv2.drawContours(frame_with_contour, [hand_contour], -1, (255, 0, 0), thickness=2)

        # 다각형 근사화를 통해 손가락 끝 개수 추정
        epsilon = 0.02 * cv2.arcLength(hand_contour, True)
        approx = cv2.approxPolyDP(hand_contour, epsilon, True)

        # 손가락 끝 개수 계산
        finger_count = max(0, len(approx) - 3)  # 손바닥을 포함한 점들을 제외하여 손가락 수 추정

        # 손가락 개수에 따른 가위, 바위, 보 판별
        if finger_count == 0:
            result_text = "Rock"
        elif finger_count == 2:
            result_text = "Scissors"
        elif finger_count >= 4:
            result_text = "Paper"

    # 작은 노이즈 제거 (모폴로지 연산)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 결과 화면에 텍스트 표시
    cv2.putText(frame_with_contour, result_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 결과 화면에 표시
    combined = np.hstack((cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), frame_with_contour))
    cv2.imshow("Hand Segmentation and Gesture Recognition", combined)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
