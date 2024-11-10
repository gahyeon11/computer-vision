import cv2
import numpy as np

# 자동차 부분만 크롭 (예제 좌표, 필요에 따라 조정 가능)
img1 = cv2.imread('img4.jpg')[10:600, 200:400]  # 자동차 부분을 크롭

# 장면 이미지 전체 불러오기
img2 = cv2.imread('img4.jpg')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# SIFT 특징점 추출
sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray2, None)

# FLANN 매칭
flann_matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
knn_match = flann_matcher.knnMatch(des1, des2, 2)

# 좋은 매칭 필터링
T = 0.7
good_match = []
for nearest1, nearest2 in knn_match:
    if (nearest1.distance / nearest2.distance) < T:
        good_match.append(nearest1)

# good_match 특징점의 위치 추출
points1 = np.float32([kp1[gm.queryIdx].pt for gm in good_match])
points2 = np.float32([kp2[gm.trainIdx].pt for gm in good_match])

# homography 계산
H, _ = cv2.findHomography(points1, points2, cv2.RANSAC)

# 첫 번째 이미지와 두 번째 이미지의 크기
h1, w1 = img1.shape[0], img1.shape[1]
h2, w2 = img2.shape[0], img2.shape[1]

# homography 적용된 위치 계산
box1 = np.float32([[0, 0], [0, h1 - 1], [w1 - 1, h1 - 1], [w1 - 1, 0]]).reshape(4, 1, 2)
box2 = cv2.perspectiveTransform(box1, H)

# 다각형 그리기
img2 = cv2.polylines(img2, [np.int32(box2)], True, (0, 255, 0), 8)

# 매칭 결과 이미지 생성
img_match = np.empty((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
cv2.drawMatches(img1, kp1, img2, kp2, good_match, img_match, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 결과 이미지 표시
cv2.imshow('Matches and Homography', img_match)

# 엔터 키를 누르면 이미지 저장
key = cv2.waitKey()
if key == 13:  # Enter 키의 ASCII 코드가 13
    cv2.imwrite('car_detection_result.png', img_match)
    print("이미지가 'car_detection_result.png'로 저장되었습니다.")

cv2.destroyAllWindows()
