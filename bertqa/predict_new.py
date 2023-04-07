import os
import sys
import pandas as pd
import json
from ast import literal_eval
cur_wd = os.getcwd()
sys.path.insert(0,'bertqa')
from tqdm import tqdm, tqdm_notebook
tqdm.pandas()
from cdqa.pipeline.cdqa_sklearn import QAPipeline
from cdqa.utils.filters import filter_paragraphs

reader_path = os.path.join(cur_wd,'bertqa/models/bert_qa.joblib')
DATA_DIR = "/Users/harshitrajgarhia/PycharmProjects/DSCI789/data"
FILTERED_DATA_DIR = DATA_DIR + os.sep + "filtered_data"


def make_paragraphs(articles,min_like_count=0,min_reply_count=0):
    def filter_paragraphs_on_likes_and_replies_count(row,min_like_count=min_like_count,min_reply_count=min_reply_count):
        paragraphs_data_list = row['paragraphs_data']
        paragraphs_list = [paragraph_data['text'] for paragraph_data in paragraphs_data_list
                           if paragraph_data['likes_count'] >= min_like_count
                           and paragraph_data['reply_count'] >= min_reply_count]

        row["paragraphs"] = paragraphs_list
        return row


    articles =  articles.apply(lambda x: filter_paragraphs_on_likes_and_replies_count(x,min_like_count,min_reply_count), axis=1)
    return articles

def fetch_cdqa_pipeline(reader_path):
    cdqa_pipeline = QAPipeline(reader=reader_path)
    return cdqa_pipeline

def build_knowledge_base(channel,ip_path,min_like_count,min_reply_count,min_length,max_length):
    filename = ip_path + os.sep + "qna_data" + os.sep + channel + "_video_qna_data_new.csv"
    df = pd.read_csv(filename, converters={'paragraphs_data': literal_eval})
    #df['count_before']  = df.paragraphs_data.str.len()
    df = make_paragraphs(df,min_like_count=min_like_count,min_reply_count=min_reply_count)
    #df['count_after']  = df.paragraphs.str.len()
    df = filter_paragraphs(df,min_length=min_length,max_length=max_length)
    return df


def predict_answer(query,cdqa_pipeline,num_answers):
    results=[]
    if not query.endswith('?'):
        query = query + '?'
    # Sending a question to the pipeline and getting prediction
    predictions = cdqa_pipeline.predict(query=query,n_predictions=num_answers)
    for i,prediction in enumerate(predictions):
        prediction = list(prediction)
        result = {'Rank':(i+1),'answer': prediction[0],'title': prediction[1],'paragraph':prediction[2],'score':prediction[3]}
        results.append(result)
    #print(results)
    return results

def get_answer(df,cdqa_pipeline, query,num_answers):
    # Fitting the retriever to the list of documents in the dataframe
    cdqa_pipeline.fit_retriever(df=df)
    results = predict_answer(query,cdqa_pipeline,num_answers)
    return results

def get_top_n_answers(row,number,cdqa_pipeline,min_like_count,min_reply_count,min_length,max_length):

    query = row['query']
    channel_list = ["fox","cnn","msnbc"]
    #channel_list = ["fox"]
    for channel in channel_list:
        with open(DATA_DIR + os.sep + "new_data" + os.sep + channel.upper() + "_merged.json", 'r') as f:
            video_dict = json.load(f)
            id_title_dict = {x:video_dict[x]['title'] for x in list(video_dict.keys())}

        #filtered_video_df = pd.read_csv(DATA_DIR+os.sep+"abortion_"+channel.upper()+".csv")
        channel_corpus_df = build_knowledge_base(channel,FILTERED_DATA_DIR,min_like_count,min_reply_count,min_length,max_length)
        cdqa_pipeline.fit_retriever(df=channel_corpus_df)
        answerlist = get_answer(channel_corpus_df, cdqa_pipeline, query, number)
        for i, answer in enumerate(answerlist):
            row[channel+"_answer_" + str(i + 1)] = answer['answer']
            row[channel+"_title_" + str(i + 1)] = id_title_dict[answer['title']]
            row[channel+"_paragraph_" + str(i + 1)] = answer['paragraph']
            row[channel+"_score_" + str(i + 1)] = answer['score']
            row[channel+"_videoID_" + str(i + 1)] = answer['title']

    return row



def get_answers_across_channels(questions_list,number,min_like_count,min_reply_count,min_length,max_length):
    cdqa_pipeline = fetch_cdqa_pipeline(reader_path)
    result_df = pd.DataFrame()
    result_df['query'] = questions_list
    result_df_across_channels = result_df.progress_apply(lambda x:get_top_n_answers(x,number,cdqa_pipeline,
                                                                                    min_like_count,min_reply_count,
                                                                                    min_length,max_length),axis=1)
    return result_df_across_channels


def get_answers_for_an_experiment(min_like_count,min_reply_count,min_length,max_length):
    questions_list = ["When does life start?",
                      "Is Abortion a murder?",
                      "is abortion legal?",
                      "is abortion against the law?",
                      "is abortion a fundamental right to a woman?",
                      ]

    result_df = get_answers_across_channels(questions_list, 10,min_like_count,min_reply_count,min_length,max_length)
    answers_columns = [column for column in result_df.columns if 'answer' in column or 'query' in column]
    result_df_answers = result_df[answers_columns]
    return result_df_answers,result_df


def save_results_of_experiments():
    writer_answer = pd.ExcelWriter(FILTERED_DATA_DIR+os.sep+"qna_results"+os.sep+"answers_across_different_filters_updated.xlsx")
    writer_answer_metadata = pd.ExcelWriter(FILTERED_DATA_DIR+os.sep+"qna_results"+os.sep+"answers_with_metadata_across_different_filters_updated.xlsx")

    # write dataframe to excel sheet named 'marks'
    # save the excel file
    min_like_count=[5,0]
    min_reply_count =[5,0]
    min_length = [30,10]
    max_length=300

    for like_count in min_like_count:
        for min_len in min_length:
            sheet_name = str(like_count)+"_LKS_OR_"+str(like_count)+"_RC_"+str(min_len)+"_MIN_"+str(max_length)+"_MAX"
            print(sheet_name)
            result_df_answers,result_df = get_answers_for_an_experiment(like_count, like_count, min_len, max_length)
            result_df_answers.to_excel(writer_answer, sheet_name,index=False)
            result_df.to_excel(writer_answer_metadata, sheet_name,index=False)

    writer_answer.save()
    writer_answer_metadata.save()





if __name__=='__main__':
    # '''
    # cdqa_pipeline = fetch_cdqa_pipeline(reader_path)
    # df = build_knowledge_base(FILTERED_DATA_DIR)
    # #get_answer(df,cdqa_pipeline,'who has been affected by the coronovirus in indian army',3)
    # a = get_answer(df,cdqa_pipeline,'When does life starts?',10)
    # '''
    #
    # questions_list = ["When does life start?",
    #                   "Is Abortion a murder?",
    #                   "is abortion legal?",
    #                   "is abortion against the law?",
    #                   "is abortion a fundamental right to a woman?",
    #                  ]
    #
    # result_df = get_answers_across_channels(questions_list, 10)
    # answers_columns = [column for column in result_df.columns if 'answer' in column or 'query' in column]
    # result_df_answers = result_df[answers_columns]
    # #result_df_answers.to_csv(FILTERED_DATA_DIR+os.sep+"qna_results"+os.sep+"qna_results_across_channels.csv",index=False)
    # #result_df.to_csv(FILTERED_DATA_DIR+os.sep+"qna_results"+os.sep+"qna_results_across_channels_metadata.csv",index=False)

    save_results_of_experiments()

# is abortion legal?
# is abortion against the constitution/law?
# is abortion a fundamental right to an American?
# is abortion be treated a healthcare?
# is obortion a RIGHT TO women?