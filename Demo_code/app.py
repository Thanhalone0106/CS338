# Python In-built packages
from pathlib import Path
import PIL
import numpy as np
from PIL import Image

import streamlit as st
import tempfile

# External packages
import streamlit as st
import cv2

# Local Modules
import settings
import helper

import segment
import torch
# Setting page layout
st.set_page_config(
    page_title="License Plate Recognition using YOLOv8",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("License Plate Recognition using YOLOv8")

# Sidebar
# st.sidebar.header("ML Model Config")

# Model Options
model_type = st.sidebar.radio(
    "Task", ['Detection'])

weights_path = 'model.h5'

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 40)) / 100
# confidence = 0.4
# Selecting Detection Or Segmentation
if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)
# elif model_type == 'Segmentation':
#     model_path = Path(settings.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
    modeltext=helper.load_model_from_json(weights_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

# st.sidebar.header("Image Config")
source_radio = st.sidebar.radio(
    "Source", settings.SOURCES_LIST)
# source_radio = IMAGE
source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
    temp_file_to_save = './temp_file_1.mp4'
    temp_file_result  = './temp_file_2.mp4'
    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
                res = model.predict(uploaded_image,
                                    conf=confidence
                                    )
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
                try:
                    image = Image.open(source_img)     
                    image.save('upload_image.png')
                    image_path = 'upload_image.png'   
                    image = cv2.imread(image_path) 
                    # file_output = "blur.png"  # Đường dẫn lưu hình ảnh đã vẽ
                    with st.expander("Detection Results"):
                        for result in res:
                            for box in result.boxes:
                                coordinates_str = str(box.xyxy)
                                coordinates_str = coordinates_str.replace("tensor([[", "").replace("]])", "")
                                # st.write(coordinates_str)
                                values = coordinates_str.split(",")
                                xmin = float(values[0].strip())
                                ymin = float(values[1].strip())
                                xmax = float(values[2].strip())
                                ymax = float(values[3].strip())
                                # st.write(xmin)
                                segment.ve_hinh_chu_nhat_hinh_anh(image,xmin, ymin,xmax,ymax) 
                                # st.write(xmin, ymin,xmax,ymax)  
                        # segment.ve_hinh_chu_nhat_hinh_anh(file_anh, 391.5214, 398.7875, 527.0744, 485.5194, file_anh) 
                        for box in boxes:
                            # st.write(box.data)
                            x=1
                        # segment.crop_image(source_img, 'output.jpg',xmin, ymin,xmax,ymax)   
                        # plate_image_path = 'output.jpg'
                        # plate = cv2.imread(plate_image_path)
                        # char_list = segment.segment_characters(plate)
                        if xmin !=' ':
                            st.image('blur.png', caption='Blur Image',use_column_width=True)
                            with open("blur.png", "rb") as file:
                                btn = st.download_button(
                                label="Download image",
                                data=file,
                                file_name="blur.jpg",
                                mime="image/png"
          )
                        # text=segment.show_results(modeltext,char_list)
                        # segment.save(char_list)
                        # st.write(text)s
                        
                        
                except Exception as ex:
                    # st.write(ex)
                    st.write("No image is uploaded yet!")
                    
elif source_radio == settings.VIDEO:
    source_video = st.sidebar.file_uploader(
        "Choose an video...", type=("mp4", "wav"))
    col1, col2 = st.columns(2)
    with col1:
        try:
            if source_video is None:
                # default_image_path = str(settings.DEFAULT_IMAGE)
                # default_image = PIL.Image.open(default_image_path)
                # st.image(default_image_path, caption="Default Image",
                #          use_column_width=True)
                x=1
            else:
                st.video('x')
        except Exception as ex: 
            # st.error("Error occurred while opening the image.")
            # st.error(ex)
            x=1

    with col2:
        if source_video is None:
            # default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            # default_detected_image = PIL.Image.open(
            #     default_detected_image_path)
            # st.image(default_detected_image_path, caption='Detected Image',
            #          use_column_width=True)
            x=1
        else:
            if st.sidebar.button('Detect Objects'):
                res = model.predict(source_video,
                                    conf=confidence
                                    )
            
# elif source_radio == settings.WEBCAM:
#     helper.play_webcam(confidence, model)

# elif source_radio == settings.RTSP:
#     helper.play_rtsp_stream(confidence, model)

# elif source_radio == settings.YOUTUBE:
#     helper.play_youtube_video(confidence, model)

else:
    st.error("Please select a valid source type!")



