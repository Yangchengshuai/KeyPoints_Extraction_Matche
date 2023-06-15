
import cv2

# 读取彩色图像
image1_color = cv2.imread('/home/ccy/code_test/img1.jpg', 1)
image2_color = cv2.imread('/home/ccy/code_test/img2.jpg', 1)

# 读取灰度图像
image1 = cv2.imread('/home/ccy/code_test/img1.jpg', 0)
image2 = cv2.imread('/home/ccy/code_test/img2.jpg', 0)

# 计算SIFT特征检测和匹配的时间
start = cv2.getTickCount()
# 创建SIFT对象
sift = cv2.SIFT_create()

# 检测关键点和计算描述子
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

# 创建FLANN匹配器
flann = cv2.FlannBasedMatcher()

# 进行特征匹配
matches = flann.knnMatch(descriptors1, descriptors2, k=2)

# 筛选匹配结果
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

end = cv2.getTickCount()
print('SIFT匹配时间:', (end - start) / cv2.getTickFrequency(), 's')

# 绘制匹配结果
result = cv2.drawMatches(image1_color, keypoints1, image2_color, keypoints2, good_matches, None)

# 保存结果
cv2.imwrite('SIFT_Matches.jpg', result)
