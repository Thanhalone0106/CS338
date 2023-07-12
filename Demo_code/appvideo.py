import streamlit as st
import cv2
import subprocess
import segment
video_data = st.file_uploader("Upload file", ['mp4','mov', 'avi'])

temp_file_to_save = './temp_file_1.mp4'

# func to save BytesIO on a drive
def write_bytesio_to_file(filename, bytesio):
    """
    Write the contents of the given BytesIO to a file.
    Creates the file or overwrites the file if it does
    not exist yet. 
    """
    with open(filename, "wb") as outfile:
        # Copy the BytesIO stream to the output file
        outfile.write(bytesio.getbuffer())

if video_data:
    # save uploaded video to disc
    write_bytesio_to_file(temp_file_to_save, video_data)

    # read it with cv2.VideoCapture(), 
    # so now we can process it with OpenCV functions
    cap = cv2.VideoCapture(temp_file_to_save)

    # grab some parameters of video to use them for writing a new, processed video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_fps = cap.get(cv2.CAP_PROP_FPS)  ##<< No need for an int
    st.write(width, height, frame_fps)
    
    # specify a writer to write a processed video to a disk frame by frame
    fourcc_mp4 = cv2.VideoWriter_fourcc(*'mp4v')
    
   
    
    ## Close video files
    
    cap.release()

    ## Reencodes video to H264 using ffmpeg
    ##  It calls ffmpeg back in a terminal so it fill fail without ffmpeg installed
    ##  ... and will probably fail in streamlit cloud
    convertedVideo = "./testh264.mp4"
    # subprocess.call(args=f"ffmpeg -y -i {temp_file_result} -c:v libx264 {convertedVideo}".split(" "))
    
    ## Show results
    col1,col2 = st.columns(2)
    col1.header("Original Video")   
    col1.video(temp_file_to_save)
    video_path = 'temp_file_1.mp4'
    output_path = 'output_video.mp4'
    model_weights = 'weights/bestYOlOV8.pt'
    # if st.sidebar.button('Detect Objects'):
    #     segment.ve_bounding_box_tren_video(video_path, output_path, model_weights)
if video_data is None:
            # default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            # default_detected_image = PIL.Image.open(
            #     default_detected_image_path)
            # st.image(default_detected_image_path, caption='Detected Image',
            #          use_column_width=True)
        x=1
else:
    if st.sidebar.button('Detect Objects'):
        x=1
        # segment.ve_bounding_box_tren_video(video_path, output_path, model_weights)