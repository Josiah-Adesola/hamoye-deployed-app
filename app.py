import numpy as np
import pickle
import spacy
import pandas as pd
import sklearn
import re
import string
import matplotlib.pyplot as plt
import nltk
from wordcloud import WordCloud 
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk import sent_tokenize

nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('wordnet')


#for feature selection
from sklearn import decomposition

import streamlit as st

#loading the saved model


rf_model = pickle.load(open("models/rf_model.pkl", 'rb'))
lr_model = pickle.load(open("models/lr_model.pkl", 'rb'))
dt_model = pickle.load(open("models/dt_model.pkl", 'rb'))

vectorizer = pickle.load(open("vectorized.pkl", 'rb'))

def clean_data(text):
    text = text.lower()  # convert all the text into lowercase
    text = text.strip()  #remove starting and trailing whitespaces
    #special_chars = re.compile('[@!#$%^&*()<>?/\|}{~:;]')
    #text = re.sub(special_chars,'', text)
    special_char_reg = '([a-zA-Z0-9]+)' + '[!"#$%&\'()*+,-./:;<=>?@\\^_`{|}~]' + '([a-zA-Z0-9]+)'
    text = re.sub(special_char_reg, ' ', text)
    text = re.sub(r'\s+', ' ', text) #remove all line formattings
    text = re.sub(r'\d+', '', text) #remove digits
    text = ''.join(c for c in text if c not in string.punctuation)   #remove pecial symbols from job titles
    return text


def lemma(text):
    word_list = nltk.word_tokenize(text) #tokenize beofre lemmatization
    lemma_output = ' '.join(WordNetLemmatizer().lemmatize(word) for word in word_list)
    return lemma_output

def main():
    st.title("Armenian Job Posting Prediction")
    st.markdown("A Project by team PP22/J609 Pyspark")
    st.header("A model for job seekers to digest online job descriptions, get keywords and incorporate them into their applications.")

    st.image("forhire.jpg")
    
    activity = ["Prediction", "NLP"]
    choice = st.sidebar.selectbox("Select Activity", activity)

    if choice == "Prediction":
        st.info("Prediction with ML")

        jobdescribe = st.text_area("Enter Job Description", "Type Here")
        all_models = ["Logistic Regression", 'Decision Trees', 'Random Forest', 'KNNeighbors']
        model =  st.selectbox("Select Model", all_models)

        # prediction_labels = [1: "IT JOB", 0: "NON-IT JOB"]

        if st.button("Classify"):
            st.info("Origin Text :: \n {}".format(jobdescribe))
            #jobdescribe = jobdescribe.toarray().reshape(1, -1)
            texts = clean_data(jobdescribe)
            # texts = lemma(texts)
            # texts = texts.split(" ")
            stop = nltk.corpus.stopwords.words('english')
            stop.extend(['armenian', 'armenia', 'job', 'title', 'position', 'location', 'responsibilities', 'application',
                'procedures', 'deadline', 'required','qualifications', 'renumeration', 'salary', 'date', 'company', 'llc'])
            # texts = text.apply(lambda x : ' '.join(x for x in x.split() if x not in stop))
            # r_text = []
            # for text in texts:
            #     if text not in stop:
            #         r_text.append(text)
            # vectoriz =  TfidfVectorizer(ngram_range=(1,1), min_df=0.05, max_df=1.0, stop_words='english')
            # x_dtm = vectoriz.fit_transform(r_text)
            # nlp = spacy.load('en_core_web_sm')
            # # Parse the sentence using the loaded 'en' model object `nlp`
            # #doc = nlp(text)
            # # x_dtm =  " ".join([token.lemma_ for token in doc])
            # df_clust = pd.DataFrame(x_dtm.toarray())
            # df_clust = df_clust.values.reshape(1, -1)
            vect = TfidfVectorizer()
            data = sent_tokenize(texts)
            new = vectorizer.transform(data)
            # new = new.transpose()
            new = new.reshape(1, -1)
            nx, ny = new.shape
            new_data = new.reshape((nx, ny))
            df_clust = pd.DataFrame(new_data, columns=vectorizer.get_feature_names())
            # df_clust.drop_duplicates(drop=True, inplace=True)

            if model == "Random Forest":
                predictor = rf_model.predict(new)
                if (prediction == 1):
                    st.success("This is an IT job")
                elif (prediction == 0):
                    st.warning("This is not an IT job")
 

            if model == "Logistic Regression":
                predictor = lr_model.predict(new)
                if (prediction == 1):
                    st.success("This is an IT job")
                elif (prediction == 0):
                    st.warning("This is not an IT job")

            if model == "Decision Trees":
                predictor = rf_model.predict(new)
                if (prediction == 1):
                    st.success("This is an IT job")
                elif (prediction == 0):
                    st.warning("This is not an IT job")
            
            # if model == "KNeighbors Classifier":
            #     predictor = rf_model.predict(vect)

    elif choice == "NLP":
            st.info("Natural Language Processing of Text")
            jobdescription = st.text_area("Job Description","Type Here")
            nlp_task = ["Word Cloud", "Tokenization","Lemmatization"]
            task_choice = st.selectbox("Choose NLP Task",["Word Cloud"])
            if st.button("Analyze"):
                st.info("Original Text::\n{}".format(jobdescription))
                nlp = spacy.load('en_core_web_sm')
            # Parse the sentence using the loaded 'en' model object `nlp`
                doc = nlp(jobdescription)
                #Tokenization using count vectorizer
                count_vect = CountVectorizer(ngram_range=(1,1))
                jobdescription = clean_data(jobdescription)
                token = count_vect.fit_transform([jobdescription])
                title_df = " ".join([token.lemma_ for token in doc])
                title_df = lemma(title_df)
                title_df = title_df.join(title_df for title_df in title_df.split() if title_df not in stop)
                temp_df =  pd.DataFrame(token.toarray(), columns=count_vect.get_feature_names())

                count_df = temp_df.apply(lambda x : x.sum())
                count_df = pd.DataFrame(count_df).reset_index()
                count_df.columns = ['Word', 'Count']
                top_jobs = count_df.sort_values(by='Count', ascending=False)
                # plot the WordCloud image to show top 50 type of demanding jobs in armenia     
                wordcloud = WordCloud(width = 1000, height = 500).generate(' '.join(top_jobs[:50].Word))
                fig = plt.figure(figsize = (8, 8), facecolor = None) 
                plt.imshow(wordcloud) 
                plt.axis("off") 
                plt.tight_layout(pad = 0) 
                plt.show()
                st.write(fig)


    # prediction = ''
    # if action == "Word Cloud":
    #     if st.button('Prediction'):
            

    # elif action == "Classification":
    #     if st.button('Prediction'):
    #         prediction = loaded_model.predict(df_clust)
    #         if (prediction == 0):
    #             st.success("This is an IT job")
    #         elif (prediction == 1):
    #             st.warning("This is not an IT job")

if __name__ == '__main__':
    main()
