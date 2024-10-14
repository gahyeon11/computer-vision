import cv2

# 비디오 파일 경로를 사용하거나 웹캠(0)을 사용하여 비디오 캡처 시작
cap = cv2.VideoCapture(0)  # '0'은 웹캠을 사용, 파일 경로를 입력할 수도 있음

# 기본 특수효과는 원본(original)으로 설정
effect = "original"

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 현재 효과에 따라 각기 다른 효과 적용
    if effect == "gray":
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif effect == "canny":
        frame = cv2.Canny(frame, 100, 200)
    elif effect == "blur":
        frame = cv2.GaussianBlur(frame, (15, 15), 0)
    elif effect == "invert":
        frame = cv2.bitwise_not(frame)
    
    # 화면에 프레임 출력
    cv2.imshow('Special Effects Video', frame)
    
    # 키보드 입력에 따른 효과 변경
    key = cv2.waitKey(30) & 0xFF
    if key == ord('g'):
        effect = "gray"  # 그레이스케일
    elif key == ord('c'):
        effect = "canny"  # 캐니 엣지 감지
    elif key == ord('b'):
        effect = "blur"  # 블러 처리
    elif key == ord('i'):
        effect = "invert"  # 색 반전
    elif key == ord('o'):
        effect = "original"  # 원본으로 되돌리기
    elif key == 27:  # ESC 키로 종료
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
