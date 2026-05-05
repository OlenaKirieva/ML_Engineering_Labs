import os

import pandas as pd
import streamlit as st
from PIL import Image


@st.cache_data  # Кешуємо, щоб не перечитувати файл при кожному кліку
def load_dataset_registry(csv_path: str):
    if not os.path.exists(csv_path):
        st.error(f"Registry file not found at {csv_path}")
        return None
    return pd.read_csv(csv_path)


def load_image(image_path: str):
    try:
        return Image.open(image_path)
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None


# import pandas as pd
# from PIL import Image
# import streamlit as st

# @st.cache_data # Кешування (Part 1: Performance)
# def load_dataset_registry(csv_path: str):
#     return pd.read_csv(csv_path)

# def get_image(image_path: str):
#     return Image.open(image_path)
