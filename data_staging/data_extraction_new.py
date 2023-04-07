import pandas as pd
import os
import json
import ast
from collections import ChainMap
import numpy as np
import string

CHANNEL = "msnbc"
DATA_DIR = "/Users/harshitrajgarhia/PycharmProjects/DSCI789/data/"
FILTERED_DATA_DIR = DATA_DIR+ os.sep + "filtered_data"
data_new = pd.read_csv(DATA_DIR+os.sep+"merged_"+CHANNEL.upper()+".csv")

filtered_video_df = data_new.groupby(['videoId'],as_index=False)['commentId', 'text', 'likeCount', 'totalReplyCount'].agg(lambda x: list(x))

def get_paragraphs_dict(row):
    dict_list = []
    dict_list_len = len(row['commentId'])
    for i in range(dict_list_len):
        dict_data = {"commentId":row["commentId"][i],
                     "text":row["text"][i],
                     "likes_count":row['likeCount'][i],
                     "reply_count":row["totalReplyCount"][i]}

        dict_list.append(dict_data)

    row["paragraphs_data"]=dict_list
    return row





filtered_video_df_with_comments_new = filtered_video_df.apply(lambda x: get_paragraphs_dict(x), axis=1)
filtered_video_df_with_comments_new = filtered_video_df_with_comments_new.rename(columns={"videoId":"id"})
filtered_video_df_with_comments_new["title"] = filtered_video_df_with_comments_new["id"]

filtered_video_df_with_comments_new.to_csv(FILTERED_DATA_DIR+os.sep+"qna_data"+os.sep+CHANNEL+"_video_qna_data_new.csv",index=False)

######################
filtered_video_df_with_comments = pd.read_csv(FILTERED_DATA_DIR+os.sep+"qna_data"+os.sep+CHANNEL+"_video_qna_data_new.csv")