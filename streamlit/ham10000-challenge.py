# SKIN LESION CLASSIFICATION

# Save this as foo.py and run it with 'streamlit run foo.py [-- script args]'.
# Alternatively, run 'python -m streamlit run foo.py'.

import streamlit as st 
import sys
from pathlib import Path
import random
from PIL import Image
import pandas as pd

root_directory = Path(__file__).resolve().parent
   
if str(root_directory) not in sys.path:
    sys.path.append(str(root_directory))
    
img_dir_rel = "./images"
img_dir = root_directory.joinpath(img_dir_rel)
img_files = list(img_dir.glob("*.jpg"))
metadata_dir_rel = "./"
metadata_dir = root_directory.joinpath(metadata_dir_rel)
metadata = pd.read_csv(metadata_dir.joinpath('labels_and_predictions.csv'))

img_path = random.choice(img_files)
img = Image.open(img_path)
img_id = Path(img_path).stem
dx = metadata[metadata['image_id'] == img_id]['dx'].values[0]
pred = metadata[metadata['image_id'] == img_id]['pred_final'].values[0]
common_name = {'nv' : 'mole', 'mel' : 'melanoma' }
  
st.markdown(f"<h1 style='text-align: center;'>HAM10000 CHALLENGE</h1>", unsafe_allow_html=True)

tab1, tab2, = st.tabs(["Challenge", "Citation"])

with tab1:

    col1, col2  = st.columns([6,6])      
    
    with col1:
        st.image(img, use_column_width=True)
        
    with col2:   

        st.session_state["score_human"] = st.session_state.get('score_human', 0)
        st.session_state["score_machine"] = st.session_state.get('score_machine', 0)
        
        st.text(f"Human  : {st.session_state.get('score_human', 0)}".upper())
        st.text(f"Machine: {st.session_state.get('score_machine', 0)}".upper())
        
        button_mel = st.button(common_name['mel'].upper(), key="button_mel")
        button_nv = st.button(common_name['nv'].upper(), key="button_nv") 
        
        if button_mel:
            st.session_state["score_human"] += dx == 'mel'
            st.session_state["score_machine"] += dx == pred
        
        if button_nv:
            st.session_state["score_human"] += dx == 'nv'
            st.session_state["score_machine"] += dx == pred                                


with tab2:
                    
    st.write("Tschandl, Philipp, 2018, 'The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions', [https://doi.org/10.7910/DVN/DBW86T](https://doi.org/10.7910/DVN/DBW86T), Harvard Dataverse, V4, UNF:6:KCZFcBLiFE5ObWcTc2ZBOA== [fileUNF].")


    st.divider()
    ''' [![Repo](https://img.shields.io/badge/GitHub-Repo-blue?logo=github)](https://github.com/tmfreiberg/HAM10000-skin-lesion-classification) ''' 
    st.markdown("<br>",unsafe_allow_html=True)
             
    
        


    

    
