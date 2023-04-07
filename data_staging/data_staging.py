import pandas as pd
import os
import json
import ast
from collections import ChainMap

CHANNEL = "cnn"
DATA_DIR = "/Users/harshitrajgarhia/Fall 2022/DSCI 789/data"
CHANNEL_DIR = DATA_DIR + os.sep + CHANNEL + "_parsed"
METADATA_DIR = CHANNEL_DIR + os.sep + "metadata"
JSON_BY_LINE_DIR = CHANNEL_DIR + os.sep + "jsonbyline"
VIDEO_DATA_DIR = DATA_DIR + os.sep + "DSCI-789-84740"

with open(VIDEO_DATA_DIR + os.sep + CHANNEL.upper() + "_video_data.json", 'r') as f:
    video_list = json.load(f)

video_id = "__7SMCAcAXs"

for video in video_list:
    if video['id'] == video_id:
        print("video found")
        video_details_dict = video
        print(video_details_dict)

video_comments_list_cleaned = []
with open(JSON_BY_LINE_DIR + os.sep + video_id + "_jsonbyline.txt") as f:
    for line in f:
        video_comments_list_cleaned.append(ast.literal_eval(line))

video_data = pd.read_csv(METADATA_DIR + os.sep + video_id + "_metadata.csv")

video_comment_df = pd.DataFrame.from_dict(video_comments_list_cleaned)

video_data['text'] = video_data['commentId'].map(dict(video_comment_df[['commentId', 'text']].values))



from emot.emo_unicode import UNICODE_EMOJI, UNICODE_EMOJI_ALIAS, EMOTICONS_EMO
from flashtext import KeywordProcessor

## formatting
all_emoji_emoticons = {**EMOTICONS_EMO,**UNICODE_EMOJI_ALIAS, **UNICODE_EMOJI_ALIAS}
all_emoji_emoticons = {k:v.replace(":","").replace("_"," ").strip() for k,v in all_emoji_emoticons.items()}

kp_all_emoji_emoticons = KeywordProcessor()
for k,v in all_emoji_emoticons.items():
    kp_all_emoji_emoticons.add_keyword(k, v)

video_data['text_cleaned'] = video_data['text'].apply(lambda x: kp_all_emoji_emoticons.replace_keywords(x))

