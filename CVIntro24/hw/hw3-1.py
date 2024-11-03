import cv2
import numpy as np
import sys

# 피부색 범위 정의 (HSV 공간에서 피부색 범위)
lower_skin = np.array([0, 20, 70], dtype=np.uint8)
upper_skin = np.array([20, 255, 255], dtype=np.uint8)

# 비디오 파일 로드
cap = cv2.VideoCapture('rock.mp4')
if not cap.isOpened():
    sys.exit("Error: Could not open video file.")

# 프레임 처리 루프
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 1. 프레임을 HSV 색상 공간으로 변환
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 2. 피부색 영역 이진화 (inRange 함수 사용)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # 3. 노이즈 제거 (모폴로지 연산)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 4. 윤곽선 검출
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5. 가장 큰 윤곽선 선택 (손의 영역으로 간주)
    if contours:
        hand_contour = max(contours, key=cv2.contourArea)

        # Convex Hull 계산 및 Convexity Defects 구하기
        hull = cv2.convexHull(hand_contour, returnPoints=False)
        if len(hull) > 3:
            defects = cv2.convexityDefects(hand_contour, hull)
            if defects is not None:
                # 손가락 끝 개수 계산
                finger_count = 0
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i, 0]
                    start = tuple(hand_contour[s][0])
                    end = tuple(hand_contour[e][0])
                    far = tuple(hand_contour[f][0])

                    # 손가락 사이의 깊이 (distance) 기준으로 손가락 개수 추정
                    a = np.linalg.norm(np.array(start) - np.array(far))
                    b = np.linalg.norm(np.array(end) - np.array(far))
                    c = np.linalg.norm(np.array(start) - np.array(end))
                    angle = np.arccos((a**2 + b**2 - c**2) / (2 * a * b)) * 57.2958

                    # 깊은 결함만 손가락 끝으로 판단
                    if angle <= 90 and d > 10000:
                        finger_count += 1

                # 가위, 바위, 보 판별
                if finger_count == 0:
                    result_text = "Rock"
                elif finger_count >= 1 and finger_count <= 2:
                    result_text = "Scissors"
                elif finger_count >= 4:
                    result_text = "Paper"
                else:
                    result_text = "Unknown"
            else:
                result_text = "Unknown"
        else:
            result_text = "Unknown"

        # 손 윤곽선 그리기
        cv2.drawContours(frame, [hand_contour], -1, (255, 0, 0), 2)
    else:
        result_text = "No Hand Detected"

    # 결과 프레임에 텍스트 표시
    cv2.putText(frame, result_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 결과 화면에 표시
    cv2.imshow("Rock Paper Scissors Detection", frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(30) & 0xFF == ord('q'):  # waitKey 값을 30으로 설정 (30 FPS 비디오)
        break

# 비디오 자원 해제
cap.release()
cv2.destroyAllWindows()
