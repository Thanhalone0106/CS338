from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import cv2
import os
import re
import matplotlib.pyplot as plt
import matplotlib.patches as patches    
import tensorflow as tf
from sklearn.metrics import f1_score
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Dropout, Conv2D
import tensorflow.keras.backend as K
from tensorflow.keras.models import Sequential, model_from_json
# Metrics for checking the model performance while training
# def f1score(y, y_pred):
#   return f1_score(y, tf.math.argmax(y_pred, axis=1), average='micro')

# def custom_f1score(y, y_pred):
#   return tf.py_function(f1score, (y, y_pred), tf.double)
# K.clear_session()
# model = Sequential()
# model.add(Conv2D(16, (22,22), input_shape=(28, 28, 3), activation='relu', padding='same'))
# model.add(Conv2D(32, (16,16), input_shape=(28, 28, 3), activation='relu', padding='same'))
# model.add(Conv2D(64, (8,8), input_shape=(28, 28, 3), activation='relu', padding='same'))
# model.add(Conv2D(64, (4,4), input_shape=(28, 28, 3), activation='relu', padding='same'))
# model.add(MaxPooling2D(pool_size=(4, 4)))
# model.add(Dropout(0.4))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(36, activation='softmax'))
# def crop_image(image_path, output_path, xmin, ymin, xmax, ymax):
#     image = Image.open(image_path)
#     cropped_image = image.crop((xmin, ymin, xmax, ymax))
#     cropped_image.save(output_path)
    
# def increase_brightness_in_dark_regions(image_path, brightness_factor, threshold):
#     # Đọc ảnh
#     image = cv2.imread(image_path)

#     # Chuyển đổi ảnh sang dạng grayscale
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Tìm kiếm vùng tối trong ảnh (dùng ngưỡng threshold)
#     _, dark_regions = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)

#     # Tăng độ sáng của các vùng tối
#     brightened_image = image.astype(np.float64)  # Chuyển đổi sang float64
#     brightened_image[dark_regions > 0] *= brightness_factor
#     brightened_image = np.clip(brightened_image, 0, 255).astype(np.uint8)  # Chuyển đổi lại thành uint8   
#     cv2.imwrite("brightened_image.jpg", brightened_image)
    
# plate_image_path = 'output.jpg'
# # Đọc ảnh bằng OpenCV
# image = cv2.imread(plate_image_path)

# # Chuyển đổi ảnh sang định dạng grayscale
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imwrite("image.jpg", gray)
# # Đọc ảnh biển số xe
# plate = cv2.imread(plate_image_path)
# def find_contours(dimensions, img):
#     # Áp dụng phép ngưỡng cường độ
#     _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

#     cntrs, _ = cv2.findContours(binary_img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     # Retrieve potential dimensions
#     lower_width = dimensions[0]
#     upper_width = dimensions[1]
#     lower_height = dimensions[2]
#     upper_height = dimensions[3]

#     # Sort contours by their x-coordinate in ascending order
#     cntrs = sorted(cntrs, key=lambda c: cv2.boundingRect(c)[0])

#     contours_list = []
#     img_res = []

#     for cntr in cntrs:
#         intX, intY, intWidth, intHeight = cv2.boundingRect(cntr)
#         if intWidth < 2 * upper_width and  2*intHeight > lower_height and intHeight <  upper_height:
#             contours_list.append((intX, intY, intX + intWidth, intY + intHeight))
#             char_copy = np.zeros((44, 24))
#             char = img[intY:intY + intHeight, intX:intX + intWidth]
#             char = cv2.resize(char, (20, 40))
#             char = cv2.subtract(255, char)
#             char_copy[2:42, 2:22] = char
#             char_copy[0:2, :] = 0
#             char_copy[:, 0:2] = 0
#             char_copy[42:44, :] = 0
#             char_copy[:, 22:24] = 0
#             img_res.append((intX, intY, intX + intWidth, intY + intHeight, char_copy))

#     # Find the y-coordinate of the center point
#     center_y = img.shape[0] // 2
#     center_y = center_y - 10

#     # Divide the characters into two rows based on the y-coordinate
#     first_row_chars = [char for char in contours_list if char[1] < center_y]
#     second_row_chars = [char for char in contours_list if char[1] >= center_y]

#     # Sort characters in each row based on x-coordinate (ascending order)
#     first_row_chars = sorted(first_row_chars, key=lambda x: x[0])
#     second_row_chars = sorted(second_row_chars, key=lambda x: x[0])

#     # Reverse the order of characters in each row
#     first_row_chars = first_row_chars[::-1]
#     second_row_chars = second_row_chars[::-1]

#     # Reverse the order of rows
#     first_row_chars = first_row_chars[::-1]
#     second_row_chars = second_row_chars[::-1]

#     # Combine the sorted characters into a single list
#     sorted_chars = first_row_chars + second_row_chars

#     # Update characters with sorted coordinates in img_res
#     sorted_chars_updated = []
#     for char in sorted_chars:
#         xmin, ymin, xmax, ymax = char
#         for i in range(len(img_res)):
#             if img_res[i][0] == xmin and img_res[i][1] == ymin and img_res[i][2] == xmax and img_res[i][3] == ymax:
#                 sorted_chars_updated.append((xmin, ymin, xmax, ymax, img_res[i][4]))
#                 break

#     print("Coordinates of each character:")
#     for char in sorted_chars_updated:
#         xmin, ymin, xmax, ymax, _ = char
#         print(f"Character at ({xmin}, {ymin}), ({xmax}, {ymax})")

#     # Remove inner characters
#     chars_to_keep = []
#     for i, char1 in enumerate(sorted_chars_updated):
#         is_inside = False
#         xmin1, ymin1, xmax1, ymax1, _ = char1
#         for j, char2 in enumerate(sorted_chars_updated):
#             if i != j:  # Không kiểm tra ký tự với chính nó
#                 xmin2, ymin2, xmax2, ymax2, _ = char2
#                 # Kiểm tra xem char1 có nằm hoàn toàn bên trong char2 hay không
#                 if xmin1 >= xmin2 and ymin1 >= ymin2 and xmax1 <= xmax2 and ymax1 <= ymax2:
#                     is_inside = True
#                     break
#         if not is_inside:
#             chars_to_keep.append(char1)

#     return [char[4] for char in chars_to_keep], chars_to_keep


# Find characters in the resulting images
# def segment_characters(image) :

#     # Preprocess cropped license plate image
#     img_lp = cv2.resize(image, (333, 75))
#     img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
#     _, img_binary_lp = cv2.threshold(img_gray_lp, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#     img_binary_lp = cv2.erode(img_binary_lp, (3,3))
#     img_binary_lp = cv2.dilate(img_binary_lp, (3,3))

#     LP_WIDTH = img_binary_lp.shape[0]
#     LP_HEIGHT = img_binary_lp.shape[1]

#     # Make borders white
#     img_binary_lp[0:10,:] = 255
#     img_binary_lp[:,0:10] = 255
#     img_binary_lp[67:35,:] = 255
#     img_binary_lp[:,320:333] = 255

#     # Estimations of character contours sizes of cropped license plates
#     dimensions = [LP_WIDTH/6,
#                        LP_WIDTH/2,
#                        LP_HEIGHT/10,
#                        2*LP_HEIGHT/3]
#     # plt.imshow(img_binary_lp, cmap='gray')
#     # plt.show()
#     cv2.imwrite('contour.jpg',img_binary_lp)

    # Get contours within cropped license plate
    # char_list, x_cntr_list = find_contours(dimensions, img_binary_lp)
    # return char_list

  # Hệ số tăng độ sáng (giá trị từ 1.0 trở lên tăng độ sáng)
# plate_image_path = 'image.jpg'
# brightness_factor =100
# # Ngưỡng để xác định vùng tối (giá trị từ 0 đến 255)
# thresholddd = 250# Gọi hàm tăng độ sáng cho vùng tối trong ảnh
# increase_brightness_in_dark_regions(plate_image_path, brightness_factor, thresholddd)
# plate_image = 'output.jpg'
# # Đọc ảnh biển số xe
# plate = cv2.imread(plate_image)

# char_list = segment_characters(plate)


# # Tạo thư mục để chứa các tệp ảnh
# os.makedirs('char_images', exist_ok=True)

# Lưu các ảnh vào thư mục
# for i, image in enumerate(char_list):
#     plt.imshow(image, cmap='gray')
#     plt.axis('off')
#     plt.savefig(f'char_images/char_image_{i}.png', bbox_inches='tight', pad_inches=0.0)
#     plt.close()


# def load_model_from_json(weights_path):
#     # Load model architecture from JSON file
#     model_json = model.to_json()
#     with open("model.json", "w") as json_file:
#         json_file.write(model_json)
        
#     json_file = open('model.json', 'r')
#     loaded_model_json = json_file.read()
#     json_file.close()
#     loaded_model = model_from_json(loaded_model_json)
#     loaded_model.load_weights(weights_path)
#     return loaded_model

# # Predicting the output
# def fix_dimension(img):
#   new_img = np.zeros((28,28,3))
#   for i in range(3):
#     new_img[:,:,i] = img
#   return new_img


# def show_results(model,char_list):
#     dic = {}
#     characters = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
#     for i, c in enumerate(characters):
#         if c == '0':
#           c=''
#         dic[i] = c

#     output = []
#     for ch in char_list:  # iterating over the characters
#         img_ = cv2.resize(ch, (28, 28), interpolation=cv2.INTER_AREA)
#         img = fix_dimension(img_)
#         img = img.reshape(1, 28, 28, 3)  # preparing image for the model
#         predictions = model.predict(img)  # predicting the probabilities
#         y_ = np.argmax(predictions)  # finding the class with the highest probability
#         character = dic[y_]
#         output.append(character)  # storing the result in a list

#     plate_number = ''.join(output)
#     return plate_number
# model=load_model_from_json("model.h5") 
# print(show_results(model))


def blur(xmin, ymin, xmax, ymax):
    image_path = 'upload_image.png'
    image = cv2.imread(image_path)
    
    # Làm tròn tọa độ số thực
    xmin_int = int(round(xmin))
    ymin_int = int(round(ymin))
    xmax_int = int(round(xmax))
    ymax_int = int(round(ymax))
    
    # Vẽ hình chữ nhật với tọa độ số nguyên
    blur_image = cv2.rectangle(image, (xmin_int, ymin_int), (xmax_int, ymax_int), (128, 128, 128), -1)
    cv2.imwrite('blur_image.jpg', blur_image)
    
def ve_hinh_chu_nhat_hinh_anh(file_anh, xmin, ymin, xmax, ymax, file_output):
    # Đọc hình ảnh
    img = Image.open(file_anh)

    # Vẽ hình chữ nhật
    fig, ax = plt.subplots()
    ax.imshow(img)
    rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=1, edgecolor='none', facecolor=(128/255, 128/255, 128/255), alpha=0.99999)
    ax.add_patch(rect)

    # Tắt hiển thị trục số
    ax.axis('off')

    # Lưu hình ảnh đã vẽ
    plt.savefig(file_output, bbox_inches='tight', pad_inches=0)

    # Hiển thị hình ảnh và hình chữ nhật
    plt.show()

# Ví dụ sử dụng
# file_anh = "upload_image.png"  # Đường dẫn đến hình ảnh cần vẽ
# file_output = "regtangle.jpg"  # Đường dẫn lưu hình ảnh đã vẽ
# ve_hinh_chu_nhat_hinh_anh(file_anh, 100.5, 100.8, 300.3, 200.7, file_output)


def draw_bounding_boxes_on_video(video_path, output_path, bbox_coordinates):
    # Đọc video
    video = cv2.VideoCapture(video_path)

    # Xác định codec và video writer để ghi video đầu ra
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Đọc từng frame trong video
    while video.isOpened():
        ret, frame = video.read()

        if not ret:
            break

        # Vẽ bounding box trên frame
        for bbox in bbox_coordinates:
            
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Ghi frame đã được vẽ bounding box vào video đầu ra
        output_video.write(frame)

        # Hiển thị video
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Giải phóng các tài nguyên
    video.release()
    output_video.release()
    cv2.destroyAllWindows()

# Sử dụng hàm để vẽ bounding box trên video
video_path = 'videos/video2.mp4'
output_path = 'output_video.mp4'
bounding_boxes = [(100, 100, 200, 200), (300, 300, 400, 400)]  # Ví dụ các bounding box

# draw_bounding_boxes_on_video(video_path, output_path, bounding_boxes)


from ultralytics import YOLO
import cv2
image_path='images/bienvang.jpg'
img_source = cv2.imread(image_path)
model = YOLO('weights/best_3.pt')
img = cv2.cvtColor(img_source, cv2.COLOR_BGR2RGB)
results = model.predict(source=img)
# Read the original image
image = cv2.imread(image_path)
for i, result in enumerate(results):
    for j, box in enumerate(result.boxes):
        # Chuỗi ký tự chứa tọa độ
        coordinates_str = str(box.xyxy)

        # Loại bỏ các ký tự không phải là số hoặc dấu chấm từ chuỗi
        print(coordinates_str)
        # coordinates_str = coordinates_str.replace("tensor([[ ").replace("]],")
        # Trích xuất các giá trị số từ chuỗi
        values = re.findall(r'\d+\.\d+', coordinates_str)
        xmin, ymin, xmax, ymax = map(float, values)
        # Assign coordinates to xmin, ymin, xmax, ymax
        ve_hinh_chu_nhat_hinh_anh(image_path,xmin,ymin,xmax,ymax,'ff.png')
        # Draw the rectangle on the image
# cv2.imshow(image)