import cv2
from matplotlib import pyplot as plt

img = cv2.imread("/home/lijun/git/image-eage-detection-learn/pic/mao.jpeg", 0)
edges = cv2.Canny(img, 100, 200)

contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, 0, (255, 0, 0), 3)

plt.subplot(131)
plt.imshow(img, cmap="gray")
plt.title('original image'), plt.xticks([]), plt.yticks([])
plt.subplot(132)
plt.imshow(edges, cmap="gray")
plt.title('edges image'), plt.xticks([]), plt.yticks([])


plt.show()
