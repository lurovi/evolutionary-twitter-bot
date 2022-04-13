import numpy as np
import pandas as pd
import argparse
import os
import re
import datetime
import collections
import nltk
import string
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import sent_tokenize, word_tokenize, TweetTokenizer
from nltk.stem import WordNetLemmatizer
nltk.download
nltk.download("wordnet")
nltk.download("stopwords")
nltk.download('averaged_perceptron_tagger')

def sorensen_dice_score(a,b):
    '''
    Compute the Sorense-Dice coefficient between two lists of tokens.
    The higher this score is, the higher is the similarity between the two provided lists.
    '''
    s1, s2 = set(a), set(b)
    intersection = float(len(s1.intersection(s2)))
    return (2*intersection) / (len(s1) + len(s2))

def preprocess_text(text, stopwords, tokenizer, lemmatizer):
    """
    This method pre-process a text throughout several steps including lowering text, deleting punctuation,
    extracting hashtag and mentioned users in case of Twitter texts, removing digits and stopwords, tokenization
    and lemmatization. Single-character words, URLS, and empty words, are also removed.
    The method returns a tuple containing a list of text tokens, a list of hashtags, a list of mentioned users, the number of URLs, and the number of digits.
    """
    if not isinstance(text,str):
        return [], [], [], 0, 0
    # Lowering and stripping/trimming text
    newText = text.lower().strip()
    # Counting the number of URLs in the current tweet
    numberOfURLs = np.sum([1 if newText.startswith("http:", i) or newText.startswith("https:", i) else 0 for i in range(0,len(newText))])
    newText = newText.replace(r"\n","    ")
    newText = newText.replace(r"\t","    ")
    newText = re.sub(r"http\S+", "    ", newText) # removing url
    newText = re.sub(r"\$\d+", "    ", newText) # removing integer price in dollars
    newText = re.sub(r"€\d+", "    ", newText) # euros
    newText = re.sub(r"£\d+", "    ", newText) # that
    newText = re.sub(r"%\d+", "    ", newText) # percentage
    newText = newText.encode("ascii", "ignore").decode() # removing \uFFFF and stuff like that
    # Deleting punctuations
    for char in (set([c for c in string.punctuation]) - set(["@","#","_","-"])).union(set(["…"])):
        newText = newText.replace(char,"    ")
    # Extracting hashtags and mentioned users
    hashtags = []
    mentionedUsers = []
    for word in newText.split():
        indexHashtag = word.find("#")
        indexMentionedUser = word.find("@")
        if indexHashtag!=-1:
            hashtags.append(word[indexHashtag:])
        if indexMentionedUser!=-1:
            mentionedUsers.append(word[indexMentionedUser:])
    newText = newText.replace("-_-","    ")
    newText = newText.replace("_-_","    ")
    newText = newText.replace("-","")
    newText = newText.replace(" _ ","    ")
    # Removing digits
    numberOfDigits = np.sum([1 if c.isdigit() else 0 for c in newText])
    newText = re.sub(r"\s[0-9]+\s","    ",newText)
    newText = re.sub(r"^[0-9]+\s","    ",newText)
    newText = re.sub(r"\s[0-9]+$","    ",newText)
    # Replacing multiple white-spaces with single white-spaces
    newText = re.sub(r"\s+"," ",newText)
    # Word tokenization and Part-of-Speech tagging
    wordTokens = nltk.pos_tag(tokenizer.tokenize(newText))
    # Lemmatization
    lemmatizedWords = []
    wordNetPoSTags = {"J": wordnet.ADJ,"N": wordnet.NOUN,"V": wordnet.VERB,"R": wordnet.ADV}
    for w, t in wordTokens:
        lemmatizedWords.append(lemmatizer.lemmatize(w,wordNetPoSTags.get(t[0].upper(),wordnet.NOUN)))
    # Removing stopwords
    # Removing single-character words
    finalWords = [w for w in lemmatizedWords if w not in stopwords and len(w)>1]
    return finalWords, hashtags, mentionedUsers, numberOfURLs, numberOfDigits

def format_twitter_bot_dataset(jsonTwitterProfile, stopwords): 
    """
    This method gets as input a Dataframe where each row corresponds to a Twitter profile consisting of a set of properties.
    The output is a Dataframe where each row corresponds to a Twitter profile with a set of hand-crafted features derived by the original ones.
    """
    chosenAttr = ["id","reputation","listed_growth_rate","favourites_growth_rate","friends_growth_rate",
              "followers_growth_rate","statuses_growth_rate",
              "screenNameLength","frequencyOfWords","frequencyOfHashtags","frequencyOfMentionedUsers",
              "frequencyOfRetweets","frequencyOfURLs",
              "words_raw_count_std","hashtag_raw_count_std",
              "mentioned_users_raw_count_std",
              "tweets_similarity_mean","stdDevTweetLength"
              ]
   
    # Type casting
    jsonTwitterProfile[["screen_name","created_at","collection_time"]]=jsonTwitterProfile[["screen_name","created_at","collection_time"]].astype(str)
    jsonTwitterProfile[["id","followers_count","friends_count","favourites_count","listed_count","statuses_count"]]=jsonTwitterProfile[["id","followers_count","friends_count","favourites_count","listed_count","statuses_count"]].astype('int64')
    
    # Stripping/trimming string columns
    for i in ["screen_name","created_at","collection_time"]:
        jsonTwitterProfile[i]=jsonTwitterProfile[i].apply(lambda x: x.strip() if isinstance(x,str) else x)
    
    # Null filling
    jsonTwitterProfile["tweet"] = jsonTwitterProfile["tweet"].apply(lambda x: x if isinstance(x, list) else [])
    
    # Initializing Hashtag and Mentioned users columns
    jsonTwitterProfile["hashtag"] = None
    jsonTwitterProfile["mentioned_users"] = None
    jsonTwitterProfile["words"] = None
    jsonTwitterProfile["hashtag"] = jsonTwitterProfile["hashtag"].astype(object)
    jsonTwitterProfile["mentioned_users"] = jsonTwitterProfile["mentioned_users"].astype(object)
    jsonTwitterProfile["words"] = jsonTwitterProfile["words"].astype(object)
    jsonTwitterProfile["hashtagFrequencies"] = None
    jsonTwitterProfile["mentioned_usersFrequencies"] = None
    jsonTwitterProfile["hashtagFrequencies"] = jsonTwitterProfile["hashtagFrequencies"].astype(object)
    jsonTwitterProfile["mentioned_usersFrequencies"] = jsonTwitterProfile["mentioned_usersFrequencies"].astype(object)
    jsonTwitterProfile["wordsFrequencies"] = None
    jsonTwitterProfile["wordsFrequencies"] = jsonTwitterProfile["wordsFrequencies"].astype(object)
    jsonTwitterProfile["retweet"] = None
    jsonTwitterProfile["retweet"] = jsonTwitterProfile["retweet"].astype(object)
    
    # Initializing Statistical Columns
    jsonTwitterProfile.loc[:,["numberOfTweets","stdDevTweetLength"]] = 0.0
    
    # Computing length of screen name
    jsonTwitterProfile["screenNameLength"] = list(map(len,jsonTwitterProfile["screen_name"]))
    
    # Tweets and description pre-processing steps with nltk
    stop_words=stopwords
    tokenizer=TweetTokenizer()
    lemmatizer=WordNetLemmatizer()
    jsonTwitterProfile = jsonTwitterProfile.set_index("id")
    iso_datetime_converter = datetime.datetime.fromisoformat
    for i in jsonTwitterProfile.index:
        
        collect_time = jsonTwitterProfile.at[i,"collection_time"]
        collect_time = collect_time.strip()+"T19:00:00"
        try:
            collect_time = iso_datetime_converter(collect_time)
        except:
            raise ValueError("Profile ID: "+str(i)+". The attribute 'collection_time' does not exhibit the correct UTC date 'YYYY-MM-DD' format.")
        collect_time = collect_time.replace(tzinfo=datetime.timezone.utc).timestamp()
        dataset_publication_year = round(collect_time/31536000 - 36.55867579908676,4) # 31536000 are the seconds in a year while 36.55867579908676 = 1152914400/31536000 where 1152914400 is the unix timestamp of 15 July 2006 (Twitter launch day). 
        
        # For each user profile, after how many years has user's profile been created since Twitter launch (15 July 2006)?
        create_at = jsonTwitterProfile.at[i,"created_at"]
        create_at = create_at.strip()
        try:
            unixT = iso_datetime_converter(create_at)
        except:
            raise ValueError("Profile ID: "+str(i)+". The attribute 'created_at' does not exhibit the correct UTC datetime ISO 8601 'YYYY-MM-DDThh:mm:ss' format.")
        unixT = unixT.replace(tzinfo=datetime.timezone.utc).timestamp()
        jsonTwitterProfile.at[i,"year_account_creation"] = round(unixT/31536000 - 36.55867579908676,4) # 31536000 are the seconds in a year while 36.55867579908676 = 1152914400/31536000 where 1152914400 is the unix timestamp of 15 July 2006  (Twitter launch day).
        
        
        # Reputation
        jsonTwitterProfile.at[i,"reputation"] = (jsonTwitterProfile.at[i,"followers_count"]/(jsonTwitterProfile.at[i,"followers_count"]+jsonTwitterProfile.at[i,"friends_count"])) if (jsonTwitterProfile.at[i,"followers_count"]+jsonTwitterProfile.at[i,"friends_count"]) != 0 else 0
        
        
        # Extracting tweets length and computing statistical measures on that length
        if len(jsonTwitterProfile.loc[i]["tweet"])>0:
            tweetLengths = np.array([len(ttt) for ttt in jsonTwitterProfile.loc[i]["tweet"] if not(ttt.startswith("RT")) and not(ttt.startswith("rt"))])
            if len(tweetLengths)>0:
                
                jsonTwitterProfile.at[i,"stdDevTweetLength"] = np.std(tweetLengths)
                
        
        tweets = jsonTwitterProfile.loc[i]["tweet"]
        
   
        hashtags = []
        mentionedUsers = []
        words = []
        tweetWordSequences = []
        reTweetWordSequences = []
        numberOfURLs, numberOfDigits = 0, 0
        # Pre-process tweets of user i
        for j in range(0,len(tweets)):
            currentTweet = tweets[j]
            tweetWords, tweetHashtags, tweetMentionedUsers, tweetNumberOfURLs, tweetNumberOfDigits = preprocess_text(currentTweet,stop_words,tokenizer,lemmatizer)
            hashtags.extend(tweetHashtags)
            mentionedUsers.extend(tweetMentionedUsers)
            numberOfURLs += tweetNumberOfURLs
            numberOfDigits += tweetNumberOfDigits
            if len(tweetWords)>0:
                if tweetWords[0]=="rt":
                    reTweetWordSequences.append(tweetWords)
                else:
                    www = []
                    for w in tweetWords:
                        if w != "rt" and not(w.startswith("#")) and not(w.startswith("@")):
                            www.append(w) 
                    words.extend(www)
                    tweetWordSequences.append(tweetWords)
        jsonTwitterProfile.at[i,"hashtag"]=hashtags
        jsonTwitterProfile.at[i,"mentioned_users"]=mentionedUsers
        jsonTwitterProfile.at[i,"words"]=words
        jsonTwitterProfile.at[i,"tweet"]=tweetWordSequences
        jsonTwitterProfile.at[i,"retweet"]=reTweetWordSequences
        jsonTwitterProfile.at[i,"numberOfURLs"] = numberOfURLs
        jsonTwitterProfile.at[i,"numberOfDigits"] = numberOfDigits
        jsonTwitterProfile.at[i,"numberOfTweets"] =len(tweetWordSequences)
        jsonTwitterProfile.at[i,"numberOfRetweets"] = len(reTweetWordSequences)
        jsonTwitterProfile.at[i,"numberOfHashtags"] = len(hashtags)
        jsonTwitterProfile.at[i,"numberOfMentionedUsers"] = len(mentionedUsers)
        jsonTwitterProfile.at[i,"numberOfWords"] = len(words)
        
        # Counting hashtags, mentioned users and words of tweets of user i
        jsonTwitterProfile.at[i,"hashtagFrequencies"] = collections.Counter(hashtags)
        jsonTwitterProfile.at[i,"mentioned_usersFrequencies"] = collections.Counter(mentionedUsers)
        jsonTwitterProfile.at[i,"wordsFrequencies"] = collections.Counter(words)
        
        
        # Setting frequency of URLs, words, hashtags, mentioned users and retweets w.r.t the number of total tweets for each user
        numbersOfTweets = jsonTwitterProfile.at[i,"numberOfTweets"]
        numbersOfRetweets = jsonTwitterProfile.at[i,"numberOfRetweets"]
        totNum = numbersOfTweets + numbersOfRetweets
        jsonTwitterProfile.at[i,"totalNumberOfTweets"] = totNum
        jsonTwitterProfile.at[i,"frequencyOfURLs"] = (jsonTwitterProfile.at[i,"numberOfURLs"]/totNum) if totNum != 0 else 0 
        jsonTwitterProfile.at[i,"frequencyOfWords"] = (jsonTwitterProfile.at[i,"numberOfWords"]/numbersOfTweets) if numbersOfTweets != 0 else 0
        jsonTwitterProfile.at[i,"frequencyOfHashtags"] = (jsonTwitterProfile.at[i,"numberOfHashtags"]/totNum) if totNum != 0 else 0
        jsonTwitterProfile.at[i,"frequencyOfMentionedUsers"] = (jsonTwitterProfile.at[i,"numberOfMentionedUsers"]/totNum) if totNum != 0 else 0
        jsonTwitterProfile.at[i,"frequencyOfRetweets"] = (jsonTwitterProfile.at[i,"numberOfRetweets"]/totNum) if totNum != 0 else 0
        jsonTwitterProfile.at[i,"frequencyOfTweets"] = (jsonTwitterProfile.at[i,"numberOfTweets"]/totNum) if totNum != 0 else 0
        
        
        # Growth ratio
        jsonTwitterProfile.at[i,"current_activity_period_in_years"] = dataset_publication_year - jsonTwitterProfile.at[i,"year_account_creation"]
        
        jsonTwitterProfile.at[i,"followers_growth_rate"] = jsonTwitterProfile.at[i,"followers_count"]/jsonTwitterProfile.at[i,"current_activity_period_in_years"]
        
        jsonTwitterProfile.at[i,"friends_growth_rate"] = jsonTwitterProfile.at[i,"friends_count"]/jsonTwitterProfile.at[i,"current_activity_period_in_years"]
        
        jsonTwitterProfile.at[i,"favourites_growth_rate"] = jsonTwitterProfile.at[i,"favourites_count"]/jsonTwitterProfile.at[i,"current_activity_period_in_years"]
        
        jsonTwitterProfile.at[i,"listed_growth_rate"] = jsonTwitterProfile.at[i,"listed_count"]/jsonTwitterProfile.at[i,"current_activity_period_in_years"]
        
        jsonTwitterProfile.at[i,"statuses_growth_rate"] = jsonTwitterProfile.at[i,"statuses_count"]/jsonTwitterProfile.at[i,"current_activity_period_in_years"]
        
        
        # Raw counts standard deviation, max, mean and median
        words_raw_count = np.array(list(jsonTwitterProfile.at[i,"wordsFrequencies"].values()))
        words_raw_count = words_raw_count[words_raw_count > 2]
        if words_raw_count.size == 0:
            jsonTwitterProfile.at[i,"words_raw_count_std"] = 0.0
            
        else:
            jsonTwitterProfile.at[i,"words_raw_count_std"] = np.std(words_raw_count)
            
        
        hashtag_raw_count = np.array(list(jsonTwitterProfile.at[i,"hashtagFrequencies"].values()))
        if hashtag_raw_count.size == 0:
            jsonTwitterProfile.at[i,"hashtag_raw_count_std"] = 0.0
            
        else:
            jsonTwitterProfile.at[i,"hashtag_raw_count_std"] = np.std(hashtag_raw_count)
            
        
        mentioned_users_raw_count = np.array(list(jsonTwitterProfile.at[i,"mentioned_usersFrequencies"].values()))
        if mentioned_users_raw_count.size == 0:
            jsonTwitterProfile.at[i,"mentioned_users_raw_count_std"] = 0.0
            
        else:
            jsonTwitterProfile.at[i,"mentioned_users_raw_count_std"] = np.std(mentioned_users_raw_count)
            
        
        
        # Average similarity of user's tweets with Jaccard index
        users_tweets = jsonTwitterProfile.at[i,"tweet"]
        similarity_scores = []
        for ii in range(len(users_tweets)-1):
            currTwt = users_tweets[ii]
            for jj in range(ii+1,len(users_tweets)):
                otherTwt = users_tweets[jj]
                similarity_scores.append(sorensen_dice_score(currTwt,otherTwt))
      
        jsonTwitterProfile.at[i,"tweets_similarity_mean"] = np.mean(similarity_scores) if len(similarity_scores)>0 else 0.0
        
        
    
    jsonTwitterProfile = jsonTwitterProfile.reset_index()
    return jsonTwitterProfile[chosenAttr]
    
    
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input",help="Absolute path of the dataset containing the data you wish to classify.",type=str,default="")
    parser.add_argument("-o","--output",help="Absolute path of the output file of this procedure (a dataset with the required hand-crafted features).",type=str,default="")
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
    types_dict = {"id":"int64","screen_name":"str","followers_count":"int64","friends_count":"int64","listed_count":"int64","statuses_count":"int64","favourites_count":"int64","created_at":"str","collection_time":"str","tweet":"list"}
    dataset = pd.read_json(input_path,orient="records",typ="frame",dtype=types_dict,convert_dates=False,precise_float=False,lines=False,encoding="utf-8",encoding_errors="replace")
    original_columns=list(dataset.columns)
    original_columns.sort(reverse=False)
    good_columns=list(types_dict.keys())
    good_columns.sort(reverse=False)
    if not(original_columns==good_columns):
        raise ValueError("Your input dataset must match the format described in the README.md file. Every JSON object in your file must have exactly the following columns: "+str(good_columns))
    derived_features = format_twitter_bot_dataset(dataset,stopwords.words())
    derived_features.to_json(output_path,orient="records",lines=False)