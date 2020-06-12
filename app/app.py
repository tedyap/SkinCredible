import streamlit as st
from typing import Dict
from PIL import Image

from pandas import np


@st.cache(allow_output_mutation=True)
def get_static_store() -> Dict:
    """This dictionary is initialized once and can be used to store the files uploaded"""
    return {}


def display_img():
    img = []
    for key in static_store.keys():
        img.append(np.array(Image.open(key)))
    st.image(img, width=250)


if __name__ == "__main__":
    static_store = get_static_store()

    st.sidebar.title('CureSkin: Facial Skin Condition Tracking')
    st.sidebar.header('Test')

    result = st.file_uploader("", type=["jpg", "jpeg"])
    if result:
        # Process you file here
        value = result.getvalue()

        # And add it to the static_store if not already in
        if not value in static_store.values():
            static_store[result] = value
            display_img()
        else:
            display_img()
            st.error("You uploaded an image twice. Please upload a different image.")
    else:
        # Clear list if the user clears the cache and reloads the page
        static_store.clear()
        st.markdown("Upload one or more images of your beautiful face here :arrow_up:")

    # if st.button("Clear file list"):
    #
    #     static_store.clear()
    #     st.markdown("Upload one or more images of your beautiful face here :arrow_up:")




