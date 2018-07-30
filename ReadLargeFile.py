#Read large file
#Read yelp data  
#Revised on 7/19/2018
import json
import pandas as pd
from pandas import DataFrame, Series
import csv,codecs, cStringIO

#The UTF8Recoder is from Python official document website
class UTF8Recoder:
    """
    Iterator that reads an encoded stream and reencodes the input to UTF-8
    """
    def __init__(self, f, encoding):
        self.reader = codecs.getreader(encoding)(f)

    def __iter__(self):
        return self

    def next(self):
        return self.reader.next().encode("utf-8")

class UnicodeReader:
    """
    A CSV reader which will iterate over lines in the CSV file "f",
    which is encoded in the given encoding.
    """

    def __init__(self, f, dialect=csv.excel, encoding="utf-8", **kwds):
        f = UTF8Recoder(f, encoding)
        self.reader = csv.reader(f, dialect=dialect, **kwds)

    def next(self):
        row = self.reader.next()
        return [unicode(s, "utf-8") for s in row]

    def __iter__(self):
        return self

#from Python document 2
class UnicodeWriter:
    """
    A CSV writer which will write rows to CSV file "f",
    which is encoded in the given encoding.
    """

    def __init__(self, f, dialect=csv.excel, encoding="utf-8", **kwds):
        # Redirect output to a queue
        self.queue = cStringIO.StringIO()
        self.writer = csv.writer(self.queue, dialect=dialect, **kwds)
        self.stream = f
        self.encoder = codecs.getincrementalencoder(encoding)()

    def writerow(self, row):
        self.writer.writerow([s.encode("utf-8") for s in row])
        # Fetch UTF-8 output from the queue ...
        data = self.queue.getvalue()
        data = data.decode("utf-8")
        # ... and reencode it into the target encoding
        data = self.encoder.encode(data)
        # write to the target stream
        self.stream.write(data)
        # empty queue
        self.queue.truncate(0)

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)


'''            
#This is for read a large file        
file_input=open('review.json','r')
business_id=[]
review_id=[]
user_id=[]
stars=[]
date=[]
text=[] #review text
useful=[]
funny=[]
cool=[]


#read data
NumberofReviews=400000

for i in range(NumberofReviews):
    line=file_input.next()
    #print line
    results=json.loads(line)
    #print '*****'
    #print results
    business_id.append(results['business_id'])
    user_id.append(results['user_id'])
    review_id.append(results['review_id'])
    useful.append(results['useful'])
    text1=results['text'].replace('\n','')
    text.append(text1)
    stars.append(results['stars'])
    cool.append(results['cool'])
    funny.append(results['funny'])
    

file_input.close()



column_names=['business_id','user_id','review_id', 'useful','stars','cool','funny','text']

with open('yelp_part.csv','wb') as result:
    
    #writer= csv.writer(result)
    #writer.writerow(column_names)
    #for i in range(NumberofReviews):
    #    writer.writerow((business_id[i],user_id[i], review_id[i], useful[i],\
    #                    stars[i], text[i]))
    
    writer=UnicodeWriter(result)
    writer.writerow(column_names)
    for i in range(NumberofReviews):
        writer.writerow((business_id[i],user_id[i], review_id[i], str(useful[i]),\
                        str(stars[i]),str(cool[i]), str(funny[i]), text[i]))

#Read a large file

'''


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

Model_Group={'LR':LogisticRegression(), 'MulNB':MultinomialNB(), 'RF':RandomForestClassifier(),
             'L_SVC': LinearSVC()}

def Pred_stars(NumofLines, InputFileName,ModelName,Fit_Model):
    #ModelName is a string indicating which model is used
    #Fit_Model=Fitting model
    number_of_rows=NumofLines
    yelp_part=pd.read_csv(InputFileName, nrows=number_of_rows)

    #pick only star rating 1 and 5
    yelp_part=yelp_part.loc[(yelp_part['stars']==1) | (yelp_part['stars']==2)]

    review_comment=yelp_part['text']
    review_stars=yelp_part['stars']
    del yelp_part

    #X_data=CountVectorizer(analyzer='word',stop_words='english').fit(review_comment)
    X_data=CountVectorizer(analyzer='word', ngram_range=(1, 1),stop_words='english').fit(review_comment)
    X_data=X_data.transform(review_comment)                       


    X_train, X_test, y_train, y_test = train_test_split(X_data, review_stars, 
                                   test_size=0.3, random_state=46, stratify=review_stars)
                                   


    #Mulnb = Fit_Model
    #Mulnb.fit(X_train, y_train)
    Fit_Model.fit(X_train, y_train)

    preds = Fit_Model.predict(X_test)

    from sklearn.metrics import confusion_matrix, classification_report

    confusion_data=confusion_matrix(y_test, preds)

    classification_data=classification_report(y_test, preds)

    #file_name='data_'+str(number_of_rows-1)+'.txt'
    file_name='data_'+str(number_of_rows-1)+ModelName+'_star1_2'+'.txt'
    with open(file_name,"w") as f:
        f.write("\n".join(" ".join(map(str, x)) for x in confusion_data))
        f.write('\n')
        f.write('\n')
        f.write(classification_data)


def main():
    num_of_lines=20001
    input_file_name='yelp_part.csv'
    ModelName=['LR','MulNB', 'RF', 'L_SVC']
    Pred_stars(num_of_lines,input_file_name, ModelName[1],Model_Group[ModelName[1]])
    
    
    
if __name__=='__main__':
    main()


        