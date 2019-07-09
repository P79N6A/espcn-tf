import cv2
import os

# path = 'result/005.jpg'
# tmp = cv2.imread(path)
#
# print(tmp)
#
# cv2.namedWindow('show005', 0)
# cv2.imshow('show005', tmp)
# cv2.waitKey(0)


# cap = cv2.VideoCapture('result/foreman.yuv')
#
# while cap.isOpened():
#     ret, frame = cap.read()
#
#     cv2.namedWindow('show11', 0)
#     cv2.imshow('show11', frame)
#
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()

file_dir = 'result/yuv/jpg'
file_list = os.listdir(file_dir)

AVATAR_STRANGER_PATH = "result/yuv/jpg"
image = os.path.join(AVATAR_STRANGER_PATH, "2.jpg")
# print(image)

#给文件改名
# for pic in file_list:
#     file_path = os.path.join(file_dir, pic)
#     print(file_path)
# # # 分离文件名和目录
# # dirname, filename = os.path.split(image)
# # # print(dirname, filename)
#     dirname, filename = os.path.split(file_path)
#     print(dirname, filename)
#     filename = filename[9:]
#     print(filename)
#     new_file = os.path.join(dirname, filename)
#     os.rename(file_path, new_file)
#
# # 改名
# new_file = os.path.join(dirname, "6.jpg")
# # print(new_file)
# os.rename(image, new_file)


file_list.sort(key=lambda x: int(x[:-4]))
print(file_list)
#
for i in range(len(file_list)):
    file_path = os.path.join(file_dir, file_list[i])
    print(file_path)
    img = cv2.imread(file_path)

    cv2.namedWindow('show_sr', 0)
    cv2.imshow("show_sr", img)
    cv2.waitKey(60)