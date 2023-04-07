#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import json
import ast
import string


# In[2]:


#Used the below keywords to filter the data
#The following keywords can be used ['baby', 'babies', 'mother', 'sarah'] but at the risk of unneccessary data
keywords = ['roewade','roe', 'wade', 'prolife', 'antiabortion', 'abortion', 
    'unborn', 'baby', 'conception', 'antiwomen', 'antiwoman','reproduction', 'fetal', 'birth'
    'fetus', 'reproduction', 'reproductive', 'embryo', 'pregnant', 
    'childbirth', 'parenthood', 'motherhood', 'pregnancy', 'fourteenth amendment', '14th amendment','trimester', 'maternal',
    'weddington', 'jane', 'wade', 'alito', 'mississippi', 'casey', 'womb']

    #Note: 'baby' -> This keyword fetched more data related to Trump on all the news comments but also has data related to roe v wade


# In[ ]:


def process(path, keywords):
    
    os.chdir(path) #Required to change the current working directory
    data = []
    count = 0
    # iterate through all file
    for file in os.listdir():
        # Check whether file is in text format or not
        if file.endswith(".txt"):
            file_path = f"{path}/{file}"
            #opening each file from directory
            with open(file_path, "rb") as data:
                for line in data:
                    count+=1
                    dict_str = line.decode('utf-8') # converting from byte class to utf-8
                    dict_str = ast.literal_eval(dict_str) #Evaluates the string can be parsed or not
                    dict_str['text'] = dict_str['text'].lower() # converts the text to lowercase
                    dict_str['text'] = dict_str['text'].translate(str.maketrans('', '', string.punctuation)) # removing punctuation 
                    if any(word in dict_str['text'].split() for word in keywords): #checking if any one of the keyword matches the string
                        data.append(dict_str)
    return pd.json_normalize(data)     


# In[ ]:


def process_video(path, keywords):
    result = []
    #Opening the file path
    with open(path, encoding ='utf-8', errors='ignore') as f:
        data = json.load(f)
        #print(len(data))
        for i in range(len(data)): #Parsing every json object
            data[i]['title'] = data[i]['title'].lower() # converts the text to lowercase
            data[i]['title'] = data[i]['title'].translate(str.maketrans('', '', string.punctuation)) # removing punctuation 
            data[i]['description'] = data[i]['description'].lower() # converts the text to lowercase
            data[i]['description'] = data[i]['description'].translate(str.maketrans('', '', string.punctuation)) # removing punctuation 
            if any(word in (data[i]['title'].split() or data[i]['description'].split()) for word in keywords): #checking if any one of the keyword matches the string in title or description
                result.append(data[i])
            
    return pd.json_normalize(result)


# In[ ]:

CHANNEL = "cnn"
DATA_DIR = "/Users/harshitrajgarhia/Fall 2022/DSCI 789/data"
CHANNEL_DIR = DATA_DIR + os.sep + CHANNEL + "_parsed"
METADATA_DIR = CHANNEL_DIR + os.sep + "metadata"
JSON_BY_LINE_DIR = CHANNEL_DIR + os.sep + "jsonbyline"
VIDEO_DATA_DIR = DATA_DIR + os.sep + "DSCI-789-84740"

cnn_path = DATA_DIR + os.sep + "cnn" + "_parsed"+os.sep + "jsonbyline"
fox_path = DATA_DIR + os.sep + "fox" + "_parsed"+os.sep + "jsonbyline"
msnbc_path = DATA_DIR + os.sep + "msnbc" + "_parsed"+os.sep + "jsonbyline"
#cnn_path = "/Users/sam/Desktop/Courses/sem3/ML_politicaldata/Project/Data/cnn_parsed/jsonbyline" #time_taken = 24m 5.9s, rows = 209966
#fox_path = "/Users/sam/Desktop/Courses/sem3/ML_politicaldata/Project/Data/fox_parsed/jsonbyline" #time_taken = 26m 23.6s, rows = 191532
#msnbc_path = "/Users/sam/Desktop/Courses/sem3/ML_politicaldata/Project/Data/msnbc_parsed/jsonbyline" #time_taken = 15m 31.1s rows = 104759

dataframe = process(msnbc_path, keywords)


# In[ ]:


cnn_video = "/Users/sam/Desktop/Courses/sem3/ML_politicaldata/Project/Data/CNN_video_data.json" #time_taken = 2.9s rows = 1117
fox_video = "/Users/sam/Desktop/Courses/sem3/ML_politicaldata/Project/Data/FOX_video_data.json" #time_taken = 2.6s rows = 432
msnbc_video = "/Users/sam/Desktop/Courses/sem3/ML_politicaldata/Project/Data/MSNBC_video_data.json" #time_taken = 5.1s rows = 381

dataframe = process_video(msnbc_video, keywords)




dataframe.to_hdf('/Users/sam/Desktop/Courses/sem3/ML_politicaldata/Project/Data/csv_files/msnbc_video.hdf', key='msnbc_video_df') # converting to hdf files


# In[ ]:


"""
Use the below keys to read the respective hdf files,

Comments_data
filename: cnn_comments.hdf ->  key: df
filename: fox_comments.hdf ->  key: fox_df
filename: msnbc_comments.hdf ->  key: msnbc_df

Filtered_Video_data
filename: cnn_video.hdf ->  key: cnn_video_df
filename: fox_video.hdf -> key: fox_video_df
filename: msnbc_video.hdf -> key: msnbc_video_df
"""
df = pd.read_hdf('/Users/sam/Desktop/Courses/sem3/ML_politicaldata/Project/Data/csv_files/fox_comments.hdf','fox_df')


df

