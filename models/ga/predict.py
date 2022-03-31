import numpy as np
import pandas as pd
import argparse
import os
import sklearn
import json
from sklearn import metrics

# ---------------------------------------------------------
# UTILITY METHODS
# ---------------------------------------------------------

def softmax_stable(x):
    '''
    This method gets as input an iterable (e.g., list, array, numpy arary) of real-value numbers and converts it into a probability distribution.
    It applies the Soft-Max Function to the input iterable.
    This is a stable version of the traditional Soft-Max Function that is able to manage effectively very large numbers.
    '''
    return np.exp(x - np.max(x))/np.exp(x - np.max(x)).sum()

def compute_binary_confusion_matrix_metrics(y_true, y_pred):
    '''
    This method gets as input an array of ground-truth labels (y_true) and an array of predicted labels (y_pred).
    It outputs a dictionary with confusion matrix values and common evaluation metrics.
    This method works for binary classification problems where the positive class is depicted with 1 while negative class is depicted with 0.
    For example, 1 for bot accounts and 0 for human accounts.
    '''
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    accuracy = (tp+tn)/(tn+fp+fn+tp)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn) # sensitivity
    specificity = tn/(tn+fp)
    fp_rate = fp/(fp+tn)
    fn_rate = fn/(fn+tp)
    f1 = (2*precision*recall)/(precision+recall)
    mcc = ( (tn*tp) - (fp*fn) )/np.sqrt( (tn+fn)*(fp+tp)*(tn+fp)*(fn+tp) )    
    roc_auc = metrics.roc_auc_score(y_true,y_pred)
    return {"total number of predictions":len(y_true),"TP":tp,"TN":tn,"FP":fp,"FN":fn,"accuracy":accuracy,"f1":f1,"mcc":mcc,"precision":precision,"recall":recall,"specificity":specificity,"fp_rate":fp_rate,"fn_rate":fn_rate,"roc_auc_score":roc_auc}


def twitter_bot_ga_model(X):
    '''
    This method gets as input a numpy matrix with N rows and D columns where N are data points while D are the features.
    It performs a binary classification for each row.
    It outputs a list with the prediction for each row (1 bot, 0 human).
    '''
    f = lambda F: int( 1.6497739285077244*F[0] + 1.5028729300118058*F[1] + 0.13125817124325725*F[2] + 1.6717674396595628*F[4] + 0.026517042164003194*F[6]  + 0.6768886250429776*F[14] +  0.7703893318740633*F[16]  <= 0.5354157875219538*F[3] + 1.3158605828957226*F[5] + 0.3703824735285971*F[7] + 0.714088296636228*F[8] + 0.6074518141987236*F[9] + 0.9329870892848713*F[10] + 1.7369084260901926*F[11] + 1.6628158914897582*F[12] + 0.12660891224677928*F[13] + 0.3983479939548097*F[15] )
    return [f(X[i]) for i in range(X.shape[0])]

# ---------------------------------------------------------
# EXECUTED CODE
# ---------------------------------------------------------

def twitter_bot_ga_model_predict_interpret(dataset):
    '''
    For each record of the input dataset, this method performs a binary classification (1 bot, 0 human).
    Moreover, each account in the dataset is associated with an ID.
    For each ID the prediction is computed alongside the components that contribute to the prediction.
    Output is a dictionary where the keys are the IDs of the accounts in the dataset and for each ID the value is another dictionary with the prediciton and other interpretability information.
    '''
    interpretations = {}
    for i in dataset.index:
        F=dataset.loc[i]
        interpretations[i] = {}
        interpretations[i]["Chuman"] = 1.6497739285077244*F[0] + 1.5028729300118058*F[1] + 0.13125817124325725*F[2] + 1.6717674396595628*F[4] + 0.026517042164003194*F[6]  + 0.6768886250429776*F[14] +  0.7703893318740633*F[16]
        interpretations[i]["Cbot"] = 0.5354157875219538*F[3] + 1.3158605828957226*F[5] + 0.3703824735285971*F[7] + 0.714088296636228*F[8] + 0.6074518141987236*F[9] + 0.9329870892848713*F[10] + 1.7369084260901926*F[11] + 1.6628158914897582*F[12] + 0.12660891224677928*F[13] + 0.3983479939548097*F[15]
        interpretations[i]["prediction"] = 1 if interpretations[i]["Chuman"] <= interpretations[i]["Cbot"] else 0
        sm = softmax_stable([interpretations[i]["Chuman"],interpretations[i]["Cbot"]])
        interpretations[i]["confidence"] = np.abs(sm[0]-sm[1])
    return interpretations
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input",help="Absolute path of the dataset containing the data you wish to classify with the correct meaningful features scaled and normalized.",type=str,default="")
    parser.add_argument("-o","--output",help="Absolute path of the output file of this procedure (a JSON file containing the results of the predictions).",type=str,default="")
    args = parser.parse_args()
    input_path = args.input
    output_path = args.output
    if input_path=="": 
        raise ValueError("Please provide a valid input file.")
    if not(os.path.exists(input_path)):
        raise ValueError("Input file does not exist. Please provide a valid dataset which exists in your file system.")
    if output_path=="":
        raise ValueError("Please provide a valid output file.")
    if os.path.exists(output_path):
        raise ValueError("Output file already exists. Please provide a file name that does not exist in your file system.")
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
    original_columns_v.sort(reverse=False)
    good_columns_v = sorted(good_columns,reverse=False)
    if not(original_columns_v==good_columns_v):
        raise ValueError("Your input dataset must match the format described in the README.md file. Every JSON object in your file must have exactly the following columns: "+str(good_columns))
    dataset = dataset.set_index("id")
    res = twitter_bot_ga_model_predict_interpret(dataset[good_columns[1:]])
    
    with open(output_path, 'w') as result_file:
        json.dump(res, result_file,  indent=6)