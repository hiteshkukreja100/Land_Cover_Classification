import streamlit as st
from PIL import Image
import warnings
from deepforest import main as deepforest_main
from deepforest import get_data
import os

warnings.filterwarnings("ignore")

@st.cache
def load_model():
    # Initialize the deepforest model
    model = deepforest_main.deepforest()
    model.use_release()
    return model

def count_trees(model, img_path):
    im = get_data("A:/Downloads/treeImage.jpg")
    cnt = model.predict_tile(im, return_plot=True, iou_threshold=0.4, patch_size=250)
    print("Predicting...")
    return cnt

def save_image(image, filename):
    image.save(filename)

def streamlit_app():
    st.title('Tree Count')

    # Upload image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display original image
        
        original_image = Image.open(uploaded_image)
        st.image(original_image, caption='Original Image', use_column_width=True)
        
        model = load_model()

        # Save the uploaded image to a temporary location
        temp_image_path = "temp_image.jpg"
        original_image.save(temp_image_path)

        imageArr = count_trees(model, temp_image_path)
        im = Image.fromarray(imageArr)
        count_of_trees = len(imageArr)
        
        # Save modified image
        save_image(im, "count.jpg")
        
        # Display saved image
        saved_image = Image.open('count.jpg')
        st.image(saved_image, caption='Counted Trees', use_column_width=True)

        # Remove the temporary image file
        os.remove(temp_image_path)

if __name__ == '__main__':
    streamlit_app()
