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

nltk.download('averaged_perceptron_tagger')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('wordnet')


#for feature selection
from sklearn import decomposition

import streamlit as st

#loading the saved model


loaded_model = pickle.load(open("rf_model.sav", 'rb'))

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
    st.header("This model should also be deployed so job seekers can quickly digest online job descriptions, get keywords, and incorporate them into their applications.")

    st.image("forhire.jpg")
    
    title = st.selectbox('Job Title', ["General Manager", "Graphic Designer", "Chief Financial Officer", "Software Developers"])
    jobdescription = st.selectbox("Job Description", ["""The Armenian Branch Office of the Open Society
Institute Assistance Foundation is seeking applications for the position
of Chief Accountant/ Finance Assistant. The Chief Accountant/ Finance
Assistant will be responsible for all transactions, connected with grant
payments, administrative expenses."
""", """"Synergy International Systems, Inc./Armenia is
currently seeking self-motivated individuals to join our quality
assurance team. The ideal candidate will meet the following basic
requirements:"
""", """"The United Nations World Food Programme is seeking an
Admin/ Finance Clerk for temporary assistance."
"""])
    requiredqual = st.selectbox("Required Qualification", ["""
    "- Bachelor's Degree; Master's is preferred;
- Excellent skills in spoken and written English and Armenian languages;
- Past English to Armenian translation and Armenian to English
translation experience;
- Good communication and public speaking skills;
- Ability to work independently and as part of a team.
REMUNERATION:  Commensurate with experience."
""", """"- Masters degree with minimum of seven years of senior project
management experience with nonprofit organizations in an international
setting; and three years of experience as chief of party managing not
less than ten staff persons;
- Experience with development of water user associations, NGO
strengthening programs, and USAID funded projects;
- Excellent ability to represent the project to donors and partners;
- Proven ability to direct all aspects of office operations, grant and
contract administration, procurement, and financial and personnel
management;
- Demonstrated diplomacy, team-orientation management, and ability to
develop and maintain collaborative, team relationships in a fast-paced
work environment;
- Excellent written and oral communications skills, and working
knowledge of computer word-processing, spreadsheet programs, and e-mail.
PREFERRED QUALIFICATIONS:  
- Previous experience in Central Asia and NIS;
- Knowledge of Russian language is a plus."
""", """"- Possession of personal vehicle, valid driver's license, and proved
5-year driving experience;
- Good communication skills;
- Good organizational skills and diligent attention to details
associated with documenting activities to maintain accurate and complete
job-related records;
- Good knowledge of logistics and working knowledge of transportation
systems.
- Written and spoken proficiency in Armenian, and Russian.
- Computer literacy, including knowledge of and experience with word
processors (MS Word), spreadsheets (Excel), databases (MS Access), and
electronic mail;
- Knowledge of, and ability to work with a variety of governmental and
non-governmental organizations;
- Mobility and desire to travel extensively;
- Willingness to work long or unusual hours/week-ends unexpectedly in
order to receive and distribute humanitarian supplies and to meet
programmatic goals and objectives;
- Willingness and ability to work in a smoke-free environment.
REMUNERATION:  Counterpart International offers competitive salaries and
benefits comparable to standards of international NGO community in
Armenia. Salary is commensurate with experience. Counterpart is an equal
opportunity organization that strives for diversity and employs
qualified personnel without regard to gender, race, physical disability,
religion, or ethnicity."
"""])
    jobrequirment =  st.selectbox("Job requirement", ["""
    "- Network monitoring and administration;
- Database administration (MS SQL 2000)."
""", """
"Consultant will develop a clear and thorough
understanding of a certain product's local consumption and in future
years consumption in neighboring countries. To do this he/she will
prepare a clear analysis of the national and regional supply and demand
of this and related consumer products. The analysis should answer the
following questions concerning:
- Supply and Demand Situation;
- Economic Analysis of Canning and/or packaging of a new product  in
Armenia;
- Market Introduction and Acceptability Procedures."
""", """
"- Testing software at all levels;
- Analyzing and reporting test results;
- Working independently with the aim of creating a test environment;
- Creating and maintaining test definitions and specifications;
- Automating test procedures and writing test automation scripts;
- Creating templates based on test results;
- Analyzing software performance and reporting data metrics;
- Developing best-case test scenarios;
- Debugging, analyzing and fixing application problems/ issues."
"""])
    
    aboutc =  st.selectbox("About Company", ["""
    "The International Research & Exchanges Board (IREX) is
a US-Based private, non-profit organization. The IREX Armenia Yerevan
office was established in 1992 and is a place in Armenia where
interested individuals can obtain up-to-date information on study,
research, and professional internship opportunities in the Unites
States.
IREX Yerevan collaborates with national government branches, local and
international NGOs and institutions of higher education in the promotion
of IREX- administered research and professional programs. The goal of
these programs is to make American academic and professional experiences
available to qualified individuals."
""", """
"The Media Diversity Institute (MDI) is a London-based
charitable organization specializing in media training. It is
implementing a three-year project in the South Caucasus, working with
the media, journalism schools and local NGOs. The project aims to create
deeper public understanding of diversity, minority groups and human
rights."
""", """
"ARQELL CJSC is a multidisciplinary manufacturing firm,
whereby its infrastructure requires diverse disciplines to arrive to the
company's paramount objective of manufacturing turnkey flexo graphic
printing machines and miscellaneous equipment used in the converting
industry."
""", """
"Interagent LLC is a distributor of several multinational
confectionary producing companies."
"""])
    company =  st.selectbox("Company",["""
    AMERIA Investment Consulting Company
""","""
International Research & Exchanges Board (IREX)
""","""
International Research & Exchanges Board (IREX)
""", """
NetCall Communications
""","""Armenia TV
"""])
    action = st.selectbox("Action", ["Word Cloud", "Classification"])
    word = ''
    col = [title, jobrequirment,requiredqual,  jobdescription, aboutc, company]
    class_data = word + ' '.join(col)
    class_df = class_data
    # class_df = class_df.apply(lambda x : clean_data(str(x)))
    # class_df_1 = class_df.apply(lambda x : lemma(x))
    class_df = clean_data(class_df)
    class_df = lemma(class_df)
    #stop word removal
    stop = nltk.corpus.stopwords.words('english')
    stop.extend(['armenian', 'armenia', 'job', 'title', 'position', 'location', 'responsibility', 'application',
             'procedure', 'deadline', 'requirement','qualification', 'renumeration', 'salary', 'date', 'company', 'llc',
             'person', 'employement', 'post', 'follow', 'resume', 'open', 'about', 'announcement', 'link', 'website',
             'organization', 'duration'])
    class_df_1 = class_df.join(class_df for class_df in class_df.split() if class_df not in stop)
    #Tokenization
    # tfidf_vect = TfidfVectorizer(ngram_range=(1,1), min_df = 0.01, max_df= 1.0, stop_words='english')
    # x_tdm = tfidf_vect.fit_transform([class_df_1])
    x_tdm = vectorizer.fit_transform(class_df_1)
    a = x_tdm.toarray()
    df_clust = pd.DataFrame(x_tdm.toarray(), columns=vectorizer.get_feature_names())

    # prediction = loaded_model.predict(df_clust)
    # if (prediction == 0):
    #     print("This is not an IT job")
    # else:
    #     print("This is an IT job")
    # return df_clust

    prediction = ''
    if action == "Word Cloud":
        if st.button('Prediction'):
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

    elif action == "Classification":
        if st.button('Prediction'):
            prediction = loaded_model.predict(df_clust)
            if (prediction == 0):
                st.success("This is an IT job")
            elif (prediction == 1):
                st.warning("This is not an IT job")

if __name__ == '__main__':
    main()
