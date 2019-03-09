import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread("/home/lijun/git/image-eage-detection-learn/pic/mao.jpeg", 1)

cur_path = "/home/lijun/git/image-eage-detection-learn/"
model_path = cur_path+"resources/model.yml"
img = np.float32(image)
img = img*(1.0/255.0)
retval = cv2.ximgproc.createStructuredEdgeDetection(model_path)
a = retval.detectEdges(img)
plt.subplot(121)
plt.imshow(image, cmap='gray')
plt.subplot(122)
plt.imshow(a, cmap='gray')
plt.show()
