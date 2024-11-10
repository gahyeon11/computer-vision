import cv2

# 스티처 객체 생성
stitcher = cv2.Stitcher.create()
images = []

# 이미지 불러오기 및 크기 조절
image_paths = [
    'hw5-0.jpg',
    'hw5-1.jpg',
    'hw5-2.jpg',
    'hw5-3.jpg',
    'hw5-4.jpg',
    'hw5-5.jpg',
    'hw5-6.jpg'
]

for path in image_paths:
    img = cv2.imread(path)
    resized_img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)  # 이미지 크기를 50%로 축소
    images.append(resized_img)

# 모든 이미지를 한 번에 stitch 수행
status, dst = stitcher.stitch(images)

if status == cv2.Stitcher_OK:
    # 결과를 표시하고 저장
    cv2.imshow('Stitched Image', dst)
    cv2.imwrite('stitch_result.jpg', dst)
    print("스티칭이 완료되었습니다. 결과가 'stitch_result.jpg'로 저장되었습니다.")
else:
    print("스티칭 실패. 상태 코드:", status)

cv2.waitKey(0)
cv2.destroyAllWindows()
