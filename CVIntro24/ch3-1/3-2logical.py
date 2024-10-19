import cv2
import sys
import numpy as np

src1=cv2.imread('lenna512.png')
src2=cv2.imread('opencv_logo256.png')

if src1 is None or src2 is None:
    sys.exit('파일을 찾을 수 없습니다.')

mask = cv2.imread('opencv_logo256_mask.png',cv2.IMREAD_GRAYSCALE)  #흰색
mask_inv = cv2.imread('opencv_logo256_mask_inv.png',cv2.IMREAD_GRAYSCALE)  #검은색

sy, sx = 0,0
# sy, sx = 100,100 #로고 이미지의 위치를 바꾸고 싶을 경우

rows,cols,channels = src2.shape
roi = src1[sy:sy+rows, sx:sx+cols]
# 레나이미지에서 해당되는 로고 이미지 만큼 잘라낸다
# cv2.imshow('roi', roi)


src1_bg = cv2.bitwise_and(roi, roi, mask=mask) # mask의 흰색(1)에 해당하는 roi는 그대로, 검정색(0)은 검정색으로
# 앤드 마스크를 하면 블랙에 해당하는 부분은 무조건 블랙, 화이트는 래나에 해당하는 부분이 나옴
# cv2.imshow('src1_bg', src1_bg)

src2_fg = cv2.bitwise_and(src2, src2, mask=mask_inv) # mask_inv의 흰색(1)에 해당하는 src2는 그대로, 검정색(0)은 검정색으로
#src2의 로고 부분만 흰색과 and 되어서 색이 나올 것
cv2.imshow('src2_fg', src2_fg)

dst = cv2.bitwise_or(src1_bg, src2_fg)
# 위 두 이미지를 or 하면 배경과 로고가 붙어서 나올 것이다. 
# cv2.imshow('dst', dst)

src1[sy:sy+rows, sx:sx+cols] = dst

pp=np.hstack((src1_bg,src2_fg, dst))
cv2.imshow('point processing - logical',pp)
# cv2.imshow('combine', src1)

cv2.waitKey()
cv2.destroyAllWindows()