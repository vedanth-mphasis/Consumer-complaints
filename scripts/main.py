import pickle
import numpy as np
import pandas as pd
import re
import nltk
import spacy

nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

with open(r'models\vectorizer', 'rb') as f:
    vectorizer = pickle.load(f)

with open(r'models\model', 'rb') as f:
    model = pickle.load(f)

topic_mapping = {
        0: 'Customer Support Complaint : time,call,day,company,phone,mortgage',
        1: 'Foreclosure Complaint : loan,modification,home,payment,foreclosure,mortgage',
        2: 'Legal Complaint : banklaw,court,chase,state,bankruptcy',
        3: 'Payment Processing Complaint : payment,mortgage,month,account,bank,amount	',
        4: 'Account Statement Complaint : fee,account,statement,service,customer,charge',
        5: 'Mortgage Closing Complaint : loan,home,closing,process,document,application',
        6: 'Refinance Complaint : credit,loan,rate,interest,report,refinance', 
        7: 'Escrow Complaint : insurance,escrow,mortgage,tax,property,company',
        8: 'Forbearance Complaint : forbearance,mortgage,plan,month,program,income',
        9: 'Fraud/Cheating Complaint : loan,document,property,home,name,fraud',
    }

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
    return texts_out

def predict_topic(text, nlp=nlp, model=model):
    text_1 = [nltk.word_tokenize(txt) for txt in text] 
    text_2 = lemmatization(text_1, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
    text_3 = vectorizer.transform(text_2)
    topic_probability_scores = model.transform(text_3)
    topic_probability_scores = np.round(topic_probability_scores, 1)
    topic_probability_scores = topic_probability_scores.tolist()
    return topic_probability_scores

def engine(df_input):

    texts = df_input['Text']
    column_names = ['Text', 'Topic 1', 'Topic 2', 'Topic 3']
    result_df = pd.DataFrame(columns=column_names)

    input_text = []
    for text in texts:
        samp = []
        samp.append(text)
        input_text.append(samp)

    for text in input_text:
        x = predict_topic(text = text)
        max1 = x[0].index(max(x[0]))
        x[0][max1] = -1
        max2 = x[0].index(max(x[0]))
        x[0][max2] = -1
        max3 = x[0].index(max(x[0]))
        row = [{'Text':text[0], 'Topic 1':topic_mapping[max1], 'Topic 2':topic_mapping[max2], 'Topic 3':topic_mapping[max3]}]
        result_df = result_df.append(row)

    return result_df

input_df = pd.read_csv('sample_data.csv')
result_df = engine(input_df)
result_df.to_csv('output.csv', index=False)