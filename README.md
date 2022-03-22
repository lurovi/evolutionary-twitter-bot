# Evolutionary Twitter Bot: Presentation of Classification Models for Twitter Bot Detection that Leverage Genetic Algorithms and Genetic Programming

In this repository we provide two classification models that can be employed for detecting bot profiles on Twitter social platform.
These models were learned using Evolutionary Computation methods such as Genetic Algorithms (GA) and Genetic Programming (GP).
They are both available in this repository as Python scripts alongside the respective training procedures.
The dataset that was adopted for training, validating and testing our models is the _TwiBot-20_ dataset that was kindly provided by Feng et al. [[1]](#1).
You can check their repository [here](https://github.com/BunsenFeng/TwiBot-20). 
We leveraged **DEAP** library by Fortin et al. [[2]](#2) in Python 3.8 to learn our models.
You can chek **DEAP** repository [here](https://github.com/DEAP/deap).

## Requirements

Everything you're going to find in this repository was developed and tested using Python 3.8.12. with the followings libraries alongside the respective versions:

|Package|Version|
|:---:|:---:|
|numpy|1.20.3|
|scipy|1.7.3|
|argparse|1.1|
|pandas|1.3.5|
|nltk|3.6.5|
|scikit-learn|1.0.2|
|deap|1.3|


## Guide

This repository is divided in two folders: preprocessing and models.
Preprocessing folder contains the script that properly formats your dataset and the scaler that was fitted to the _TwiBot-20_ training set that you have to use in order to transform your data.
Models folder is divided into two sub-folders: ga and gp.
They contain the respective classification models for Twitter Bot Detection.

### Preprocessing

First of all, you must ensure that the dataset containing the data you wish to classify is formatted in the way our algorithms expect it to be.
Your dataset must be a JSON file with UTF-8 encoding that contains an array of JSON objects, where each JSON object represent a Twitter account that should be classified either as bot (label 1) or human (label 0).
Each JSON object must have the following format:

|Attribute Name|Attribute Type|
|:---:|:---:|
|id|int64|
|screen_name|string|
|followers_count|int64|
|friends_count|int64|
|listed_count|int64|
|favourites_count|int64|
|statuses_count|int64|
|created_at|string|
|collection_time|string|
|tweet|list of strings|

You can check below an example of JSON object correctly formatted:


	{
	"id":10012,
	"screen_name":"example01",
	"followers_count":100,
	"friends_count":400,
	"listed_count":50,
	"favourites_count":25,
	"statuses_count":125,
	"created_at":"2022-03-21T12:30:25",
	"collection_time":"2022-03-27",
	"tweet":["this is an example of tweet!!","RT this is an example of retweet","rt Just ANOTHER example of retweet","RT @user001 another retweet","rt @user002 YET another retweet","Finally a normal tweet!!!! :-D"]
	}


All the fields of each JSON object are mandatories.
In case you don't have any tweet for a given account, in the corresponding JSON object the "tweet" field must be associated with an empty list.
For example:


	{
	"id":10013,
	"screen_name":"example02",
	"followers_count":100,
	"friends_count":400,
	"listed_count":50,
	"favourites_count":25,
	"statuses_count":125,
	"created_at":"2022-03-21T08:05:07",
	"collection_time":"2022-03-27",
	"tweet":[]
	}


Clearly, the more tweets you have for a given account, the more the prediction is reliable.
Having no tweets for a given profile means that the prediction is going to be unreliable since it won't leverage semantic information of the profile at hand.
A single tweet is represented by a normal string containing tweet text.
In particular, retweets correspond to strings that begin with either rt or RT followed by at least one empty space.
The "collection_time" field is a string representing an UTC date in "YYYY-MM-DD" format and it should approximately indicate when the meta-data associated to a given account was collected.
With meta-data, we're referring to the following attributes: "followers_count", "friends_count", "listed_count", "favourites_count", and "statuses_count".
The "collection_time" attribute is important because it enables the algorithm to figure out the age of a given account at the moment of data collection.
The other attributes have the same meaning of the corresponding properties that are retrievable using Twitter API.
Check the docs [here](https://developer.twitter.com/en/docs/twitter-api/v1/data-dictionary/object-model/user).
In particular, the "created_at" attribute is a string representing an UTC datetime compliant to ISO 8601 "YYYY-MM-DDThh:mm:ss" format.
Two examples of "created_at" string with the correct format are the following: "2022-03-21T08:05:07", "2022-03-21T21:15:09".
In the preprocessing folder you can additionally find an example of properly formatted dataset with some fake data inside (example.json).
<br/>
Once you have your dataset formatted this way, you can preprocess your dataset while transforming it with the proper scaler.
In the preprocessing folder, download and execute format_preprocess.py giving your dataset as input while defining the name for your output file.
Example of execution:


	python format_preprocess.py --input D:\Users\somebody\dataset.json --output D:\Users\somebody\dataset_formatted.json


Now, you should have obtained your dataset with all the features that are meaningful for our classification models.
Now, download and execute scale_transform.py in order to apply the scaler to this dataset.


	python scale_transform.py --input D:\Users\somebody\dataset_formatted.json --output D:\Users\somebody\dataset_scaled.json --scaler D:\Users\somebody\scaler.pkl


You can find the scaler inside the preprocessing folder as well.
<br/>
Now you're ready to perform the predictions.

### Classification

In the folder ga you can find predict.py that performs the prediction using the classification model discovered using Genetic Algorithms.
In the folder gp you can find predict.py that performs the prediction using the classification model discovered using Genetic Programming.
Example of execution:


	python predict.py --input D:\Users\somebody\dataset_scaled.json --output D:\Users\somebody\predictions.json


The output file generated by predict.py is a JSON file that contains for each Twitter profile ID the predicted category (0 for humans, 1 for bot accounts).
You should also find in the output file other information concealing prediction interpretability.
This file is a JSON file where the keys correspond to account IDs.
For each account ID in this JSON file you can find a JSON object with the prediction and other information about components that contribute to the predicted value.
Furthermore, for each of the two described approaches, you can find a train.py file with the **DEAP** training procedure that was executed in order to obtain the discovered model.

## References

<a id="1">[1]</a>
Feng, Shangbin and Wan, Herun and Wang, Ningnan and Li, Jundong and Luo, Minnan. "TwiBot-20: A Comprehensive Twitter Bot Detection Benchmark". In: Proceedings of the 30th ACM International Conference on Information & Knowledge Management. Series: CIKM \'21. Pages: 4485–4494 (2021).
<br/>
<a id="2">[2]</a>
Fortin, F\'elix-Antoine and De Rainville, Francois-Michel and Gardner, Marc-Andr\'e and Parizeau, Marc and Gagn\'e, Christian. "DEAP: Evolutionary Algorithms Made Easy". Journal of Machine Learning Research. Vol. 13. Pages: 2171-2175 (2012).

















