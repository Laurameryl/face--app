import  streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from datetime import datetime

# Set page configuration - FIXED SYNTAX
st.set_page_config(
    page_title="Face Detection App",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # App title and description
    st.title("Face Detection using Viola-Jones Algorithm")
    st.markdown("""
    This app detects faces in images using the Viola-Jones algorithm. 
    Upload an image, adjust the detection parameters if needed, and see the results!
    """)
    
    # Sidebar for instructions and parameters
    with st.sidebar:
        st.header("Instructions")
        st.markdown("""
        1. Upload an image using the file uploader
        2. Adjust detection parameters if needed:
           - **Scale Factor**: How much the image size is reduced at each scale (1.01-1.5)
           - **Min Neighbors**: How many neighbors each candidate rectangle should have (1-10)
           - **Rectangle Color**: Choose a color for the face bounding boxes
        3. View the results
        4. Download the processed image if desired
        """)
        
        st.header("Parameters")
        scale_factor = st.slider(
            "Scale Factor", 
            min_value=1.01, 
            max_value=1.5, 
            value=1.1, 
            step=0.01,
            help="Parameter specifying how much the image size is reduced at each image scale. Lower values increase detection accuracy but are slower."
        )
        
        min_neighbors = st.slider(
            "Min Neighbors", 
            min_value=1, 
            max_value=10, 
            value=5,
            help="Parameter specifying how many neighbors each candidate rectangle should have to retain it. Higher values reduce false positives but might miss some faces."
        )
        
        rect_color = st.color_picker(
            "Rectangle Color", 
            "#00FF00",
            help="Select a color for the bounding boxes around detected faces"
        )
        
        # Convert hex color to BGR for OpenCV
        hex_color = rect_color.lstrip('#')
        rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        bgr_color = (rgb_color[2], rgb_color[1], rgb_color[0])
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"],
        help="Select an image file to process (JPG, JPEG, or PNG)"
    )
    
    if uploaded_file is not None:
        # Read the image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image = image.copy()
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Load the cascade classifier
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors
        )
        
        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), bgr_color, 2)
        
        # Convert to RGB for display
        image_rgb_with_boxes = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image_rgb, use_column_width=True)
        
        with col2:
            st.subheader("Processed Image")
            st.image(image_rgb_with_boxes, use_column_width=True)
            st.write(f"Number of faces detected: **{len(faces)}**")
        
        # Save image functionality
        st.subheader("Save Processed Image")
        if st.button("Save Image to Device"):
            # Create a directory for saved images if it doesn't exist
            if not os.path.exists("saved_images"):
                os.makedirs("saved_images")
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"saved_images/face_detection_{timestamp}.jpg"
            
            # Save the image
            cv2.imwrite(filename, image)
            st.success(f"Image saved successfully as {filename}")
            
            # Offer download link
            with open(filename, "rb") as file:
                btn = st.download_button(
                    label="Download Image",
                    data=file,
                    file_name=f"face_detection_{timestamp}.jpg",
                    mime="image/jpeg"
                )
    
    else:
        # Display placeholder images when no file is uploaded
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image("https://via.placeholder.com/500x300?text=Upload+an+image", use_column_width=True)
        
        with col2:
            st.subheader("Processed Image")
            st.image("https://via.placeholder.com/500x300?text=Processed+image+will+appear+here", use_column_width=True)

if __name__ == "__main__":
    main()