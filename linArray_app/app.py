import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import alignment_functions as af
#from PIL import Image #to remove

st.set_page_config(layout='wide')
st.title( 'Linear Array Browzer' )
st.header( 'Visualizing combined behavioral and neural data' )
#st.subheader( 'Plexon Linear Arrays aligned with MonkeyLogic behavioral data' )
st.subheader( 'for the McPeek lab' ) 

st.markdown("---")

st.sidebar.header("Visualization Controls")
st.sidebar.markdown("---")
# import the data
data_pkls = {'T35_083018_t1':"T35_083018_t1_delayedSaccade.pkl",
             'T35_083018_t2':"T35_083018_t2_delayedSaccade.pkl",
             'T35_083018_t3':"T35_083018_t3_delayedSaccade.pkl"}
catSearch_pkls = {'T35_083018_t2': "T35_083018_t2_catSearchPHYSIOL.pkl",
             'T35_083018_t3': "T35_083018_t3_catSearchPHYSIOL.pkl"}
ds180_pkls = {'T35_083018_t1': "T35_083018_t1_delayedSaccade180RFInt.pkl",
             'T35_083018_t2': "T35_083018_t2_delayedSaccade180RFInt.pkl"}
data_tasks = {'T35_083018_t1':['delayedSaccIntervals', 'delayedSacc180RFInt'],
             'T35_083018_t2':['delayedSaccIntervals', 'catSearchPHYSIOLIntervals', 'delayedSacc180RFInt'],
             'T35_083018_t3':['delayedSaccIntervals', 'catSearchPHYSIOLIntervals']}

recordings = list(data_pkls.keys())
st.sidebar.write('Select a Track, Task & Channel:')
linArray_pick = st.sidebar.selectbox( "Select Track:", recordings )
linArray_upload = st.sidebar.file_uploader( 'Optional: Select a local linArray.pkl file' )
if linArray_upload is not None:
    linArray_df = pd.read_pickle( linArray_upload )
else:
    #st.stop()
    linArray_df = pd.read_pickle( data_pkls[ linArray_pick ] )

#dropdown menus to select Track, Channel & Task
tasks = data_tasks[linArray_pick]
SPK_col = [col for col in linArray_df if col.startswith('SPK')]
time_axis_limits_ms = [ -50, 150 ]

selected_task = st.sidebar.selectbox( "Select Task", tasks ) 
selected_channel = st.sidebar.selectbox( "Select Channel", SPK_col )

st.sidebar.markdown("---")
st.sidebar.write('Control alignment time windows:')
align_min = st.sidebar.number_input( label = 'alignment min: ', min_value = -200.0, max_value = 0.0, value = -50. )
align_max = st.sidebar.number_input( label = 'alignment max', min_value = 0. , max_value = 1500. , value = 150. )

time_axis_limits_ms = [ align_min, align_max ]
DS_saccadeOnset_ts = 'saccade_onset'
DS_stimOn_ts = 'bhv_code40'

empty_left, contents, empty_right = st.columns([1, 8, 1])
with empty_left:
    pass
with contents:
    if selected_task == 'delayedSaccIntervals':
        combinedalign = af.eyespike_dualAligned( linArray_df, selected_channel, DS_stimOn_ts, DS_saccadeOnset_ts, time_axis_limits_ms, 0  )
        st.pyplot( combinedalign )
    elif selected_task == 'delayedSacc180RFInt':
        ds180_df = pd.read_pickle( ds180_pkls[linArray_pick] )
        ds180 = af.eyespike_dualAligned_180( ds180_df, selected_channel, 'bhv_code40', 'saccade_onset', time_axis_limits_ms, 0  )
        st.pyplot( ds180 )
    elif selected_task == 'catSearchPHYSIOLIntervals':
        catSearch_df = pd.read_pickle( catSearch_pkls[linArray_pick] )
        catSearch = af.eyespike_stimOnAligned( catSearch_df, selected_channel, 'bhv_code40', time_axis_limits_ms, 0  )
        st.pyplot( catSearch )
    else:
        pass
        #image = Image.open('obi-wan.jpg')
        #st.image(image, caption='This is not the task you are looking for')
with empty_right:
    pass