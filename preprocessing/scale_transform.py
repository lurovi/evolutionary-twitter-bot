import numpy as np
import pandas as pd
import argparse
import os
import sklearn
import pickle
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, PowerTransformer


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input",help="Absolute path of the dataset containing the data you wish to classify with the correct meaningful features.",type=str,default="")
    parser.add_argument("-o","--output",help="Absolute path of the output file of this procedure (a dataset with the required hand-crafted features properly scaled and normalized).",type=str,default="")
    parser.add_argument("-s","--scaler",help="Absolute path of the .pkl file containing the scaler.",type=str,default="")
    args = parser.parse_args()
    input_path = args.input
    output_path = args.output
    fitted_scaler = args.scaler
    if input_path=="": 
        raise ValueError("Please provide a valid input file.")
    if not(os.path.exists(input_path)):
        raise ValueError("Input file does not exist. Please provide a valid dataset which exists in your file system.")
    if output_path=="":
        raise ValueError("Please provide a valid output file.")
    if os.path.exists(output_path):
        raise ValueError("Output file already exists. Please provide a file name that does not exist in your file system.")
    if fitted_scaler=="":
        raise ValueError("Please provide a valid scaler file.")
    if not(os.path.exists(fitted_scaler)):
        raise ValueError("Scaler file does not exist. Please provide the valid scaler after downloading it from the repository.")
    dataset = pd.read_json(input_path,orient="records",typ="frame",dtype=True,convert_dates=False,precise_float=False,lines=False,encoding="utf-8",encoding_errors="replace")
    original_columns=dataset.columns
    original_columns_v = [original_columns[i] for i in range(len(original_columns))]
    good_columns=["id","reputation","listed_growth_rate","favourites_growth_rate","friends_growth_rate",
              "followers_growth_rate","statuses_growth_rate",
              "screenNameLength","frequencyOfWords","frequencyOfHashtags","frequencyOfMentionedUsers",
              "frequencyOfRetweets","frequencyOfURLs",
              "words_raw_count_std","hashtag_raw_count_std",
              "mentioned_users_raw_count_std",
              "tweets_similarity_mean","stdDevTweetLength"
              ]
    if not(original_columns_v==good_columns):
        raise ValueError("Your input dataset must match the output format of format_preprocess.py. Every JSON object in your file must have exactly the following columns in the following order: "+str(["id","reputation","listed_growth_rate","favourites_growth_rate","friends_growth_rate","followers_growth_rate","statuses_growth_rate","screenNameLength","frequencyOfWords","frequencyOfHashtags","frequencyOfMentionedUsers","frequencyOfRetweets","frequencyOfURLs","words_raw_count_std","hashtag_raw_count_std","mentioned_users_raw_count_std","tweets_similarity_mean","stdDevTweetLength"]))
    dataset = dataset.set_index("id")
    all_index = dataset.index.tolist()
    pickle_off = open(fitted_scaler, 'rb')
    scaler_loaded = pickle.load(pickle_off)
    pickle_off.close()
    dataset_transformed = scaler_loaded.transform(dataset.values)
    df_dataset_transformed = pd.DataFrame(data=dataset_transformed, columns = ["reputation","listed_growth_rate","favourites_growth_rate","friends_growth_rate","followers_growth_rate","statuses_growth_rate","screenNameLength","frequencyOfWords","frequencyOfHashtags","frequencyOfMentionedUsers","frequencyOfRetweets","frequencyOfURLs","words_raw_count_std","hashtag_raw_count_std","mentioned_users_raw_count_std","tweets_similarity_mean","stdDevTweetLength"])
    df_dataset_transformed.insert(loc=0,column="id",value=all_index,allow_duplicates=False)
    df_dataset_transformed.to_json(output_path,orient="records",lines=False)