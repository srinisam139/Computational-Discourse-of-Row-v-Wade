import os
import sys
import pandas as pd
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

def build_knowledge_base(channel,ip_path):
    filename = ip_path + os.sep + "qna_data" + os.sep + channel + "_video_qna_data.csv"
    df = pd.read_csv(filename, converters={'paragraphs_data': literal_eval})
    df['count_before']  = df.paragraphs_data.str.len()
    df = make_paragraphs(df)
    df['count_after']  = df.paragraphs.str.len()
    df = filter_paragraphs(df)
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

def get_top_n_answers(row,number,cdqa_pipeline):
    query = row['query']
    channel_list = ["fox","cnn","msnbc"]
    #channel_list = ["fox"]
    for channel in channel_list:
        filtered_video_df = pd.read_hdf(FILTERED_DATA_DIR + os.sep + "csv_files" + os.sep + channel + '_video.hdf', channel + '_video_df')
        title_id_dict = dict(filtered_video_df[['title', 'id']].values)
        channel_corpus_df = build_knowledge_base(channel,FILTERED_DATA_DIR)
        cdqa_pipeline.fit_retriever(df=channel_corpus_df)
        answerlist = get_answer(channel_corpus_df, cdqa_pipeline, query, number)
        for i, answer in enumerate(answerlist):
            row[channel+"_answer_" + str(i + 1)] = answer['answer']
            row[channel+"_title_" + str(i + 1)] = answer['title']
            row[channel+"_paragraph_" + str(i + 1)] = answer['paragraph']
            row[channel+"_score_" + str(i + 1)] = answer['score']
            row[channel + "_id_" + str(i + 1)] = title_id_dict[answer['title']]
    return row



def get_answers_across_channels(questions_list,number):
    cdqa_pipeline = fetch_cdqa_pipeline(reader_path)
    result_df = pd.DataFrame()
    result_df['query'] = questions_list
    result_df_across_channels = result_df.progress_apply(lambda x:get_top_n_answers(x,number,cdqa_pipeline),axis=1)
    return result_df_across_channels

if __name__=='__main__':
    '''
    cdqa_pipeline = fetch_cdqa_pipeline(reader_path)
    df = build_knowledge_base(FILTERED_DATA_DIR)
    #get_answer(df,cdqa_pipeline,'who has been affected by the coronovirus in indian army',3)
    a = get_answer(df,cdqa_pipeline,'When does life starts?',10)
    '''

    questions_list = ["When does life start?",
                      "Is Abortion a murder?",
                      "is abortion legal?",
                      "is abortion against the law?",
                      "is abortion a fundamental right to a woman?",
                     ]

    result_df = get_answers_across_channels(questions_list, 5)
    answers_columns = [column for column in result_df.columns if 'answer' in column or 'query' in column]
    result_df_answers = result_df[answers_columns]
    # result_df_answers.to_csv(FILTERED_DATA_DIR+os.sep+"qna_results"+os.sep+"qna_results_across_channels.csv",index=False)
    # result_df.to_csv(FILTERED_DATA_DIR+os.sep+"qna_results"+os.sep+"qna_results_across_channels_metadata.csv",index=False)

# is abortion legal?
# is abortion against the constitution/law?
# is abortion a fundamental right to an American?
# is abortion be treated a healthcare?
# is obortion a RIGHT TO women?