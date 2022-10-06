import cv2
file_path = 'E:\digital_image_processing\datasets\\tongji\\ROI\\session1\\00001.bmp'
img = cv2.imread(file_path)  # 读取图片信息

# sp = img.shape[0:2]     #截取长宽啊
sp = img.shape  # [高|宽|像素值由三种原色构成]
print(sp)