# Stackoverflow-tag-predictor

Description

Stack Overflow is the largest, most trusted online community for developers to learn, share their programming knowledge, and build their careers.

Stack Overflow is something which every programmer use one way or another. Each month, over 50 million developers come to Stack Overflow to learn, share their knowledge, and build their careers. It features questions and answers on a wide range of topics in computer programming. The website serves as a platform for users to ask and answer questions, and, through membership and active participation, to vote questions and answers up or down and edit questions and answers in a fashion similar to a wiki or Digg. As of April 2014 Stack Overflow has over 4,000,000 registered users, and it exceeded 10,000,000 questions in late August 2015. Based on the type of tags assigned to questions, the top eight most discussed topics on the site are: Java, JavaScript, C#, PHP, Android, jQuery, Python and HTML.

Problem Statemtent

Suggest the tags based on the content that was there in the question posted on Stackoverflow.

Source: https://www.kaggle.com/c/facebook-recruiting-iii-keyword-extraction/

Data Overview 

Refer: https://www.kaggle.com/c/facebook-recruiting-iii-keyword-extraction/data

All of the data is in 2 files: Train and Test.

Train.csv contains 4 columns: Id,Title,Body,Tags.

Test.csv contains the same columns but without the Tags, which you are to predict.

Size of Train.csv - 6.75GB

Size of Test.csv - 2GB

Number of rows in Train.csv = 6034195


Data Field Explaination

Dataset contains 6,034,195 rows. The columns in the table are:

Id - Unique identifier for each question

Title - The question's title

Body - The body of the question

Tags - The tags associated with the question in a space-seperated format (all lowercase, should not contain tabs '\t' or ampersands '&')

Type of Machine Learning Problem 

It is a multi-label classification problem

Performance metric

As it is a multi label classification normal  F1 score wont work we need something else thT gets F1 score in this type of output 

thts why micro and macro F1 SCORE

Micro-Averaged F1-Score (Mean F Score) : The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal. The formula for the F1 score is:

F1 = 2 * (precision * recall) / (precision + recall)

In the multi-class and multi-label case, this is the weighted average of the F1 score of each class. 

'Micro f1 score': 

Calculate metrics globally by counting the total true positives, false negatives and false positives. This is a better metric when we have class imbalance. 

IN MICRO fI score we consider occurence of label into consideration as tags that occur less tend to have lower micro f1 score 

'Macro f1 score': 

Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account. 

Macro F1 score is avg of all of the F1 score of tags available .

Exploratory Data Analysis 

Data Loading and Cleaning 

Using Pandas with SQLite to Load the data

This time we are using sqlite to read our data instead of simply reading it from .csv file.

What we will do is we will load csv file to sqlite db, connect with that db and with the use sql queries we will use data from db.

```python
#Creating db file from csv
if not os.path.isfile('train.db'):
    start = datetime.now()
    disk_engine = create_engine('sqlite:///train.db')
    start = dt.datetime.now()
    chunksize = 180000
    j = 0
    index_start = 1
    #this loop takes all data from train.csv file and appends it to train.db
    #chunksize = The parameter essentially means the number of rows to be read into a dataframe at any single time in order to fit into the local memory
    for df in pd.read_csv('Train.csv', names=['Id', 'Title', 'Body', 'Tags'], chunksize=chunksize, iterator=True, encoding='utf-8', ):
        df.index += index_start
        j+=1
        print('{} rows'.format(j*chunksize))
        df.to_sql('data', disk_engine, if_exists='append')
        index_start = df.index[-1] + 1
```

Counting the number of rows 

```python
if os.path.isfile('train.db'):
    start = datetime.now()
    con = sqlite3.connect('train.db')
    num_rows = pd.read_sql_query("""SELECT count(*) FROM data""", con)
    #Always remember to close the database
    print("Number of rows in the database :","\n",num_rows['count(*)'].values[0])
    con.close()
    
    Number of rows in the database : 
     6034196
```

Checking for duplicates

```python
if os.path.isfile('train.db'):
    start = datetime.now()
    con = sqlite3.connect('train.db')
    df_no_dup = pd.read_sql_query('SELECT Title, Body, Tags, COUNT(*) as cnt_dup FROM data GROUP BY Title, Body, Tags', con)
    con.close()
    
    print("number of duplicate questions :", num_rows['count(*)'].values[0]- df_no_dup.shape[0])
    
    number of duplicate questions : 1827881
```
    
number of times each question appeared in our database

```python
df_no_dup.cnt_dup.value_counts()

1    2656284
2    1272336
3     277575
4         90
5         25
6          5
Name: cnt_dup, dtype: int64
```

```python
df_no_dup["tag_count"] = df_no_dup["Tags"].apply(lambda text: len(text.split(" ")))
# adding a new feature number of tags per question
df_no_dup.head()

```
```python
# distribution of number of tags per question
df_no_dup.tag_count.value_counts()

3    1206157
2    1111706
4     814996
1     568298
5     505158
Name: tag_count, dtype: int64
```

```python
#Creating a new database with no duplicates
#query to ask how we geting db with no duplicates 
if not os.path.isfile('train_no_dup.db'):
    disk_dup = create_engine("sqlite:///train_no_dup.db")
    no_dup = pd.DataFrame(df_no_dup, columns=['Title', 'Body', 'Tags'])
    no_dup.to_sql('no_dup_train',disk_dup)
```

```python
#This method seems more appropriate to work with this much data.
#creating the connection with database file.
if os.path.isfile('train_no_dup.db'):
    #start = datetime.now()
    con = sqlite3.connect('train_no_dup.db')
    tag_data = pd.read_sql_query("""SELECT Tags FROM no_dup_train""", con)
    #print(tag_data)
    #Always remember to close the database
    con.close()

    # Let's now drop unwanted column.
    tag_data.drop(tag_data.index[0], inplace=True)
    print(tag_data)
    #Printing first 5 columns from our data frame
    #tag_data.head()
    #print("Time taken to run this cell :", datetime.now() - start)
else:
    print("Please download the train.db file from drive or run the above cells to genarate train.db file")
```

Analysis of Tags 

Total number of unique tags 

```python
# Importing & Initializing the "CountVectorizer" object, which 
#is scikit-learn's bag of words tool.

#by default 'split()' will tokenize each tag using space.
vectorizer = CountVectorizer(tokenizer = lambda x: x.split())
# fit_transform() does two functions: First, it fits the model
# and learns the vocabulary; second, it transforms our training data
# into feature vectors. The input to fit_transform should be a list of strings.
tag_dtm = vectorizer.fit_transform(tag_data['Tags'])
```

```python
print("Number of data points :", tag_dtm.shape[0])
print("Number of unique tags :", tag_dtm.shape[1])

Number of data points : 4206314
Number of unique tags : 42048
```

```python
#'get_feature_name()' gives us the vocabulary.
tags = vectorizer.get_feature_names()
#Lets look at the tags we have.
print("Some of the tags we have :", tags[:10])

Some of the tages we have : ['.a', '.app', '.asp.net-mvc', '.aspxauth', '.bash-profile', '.class-file', '.cs-file', '.doc', '.drv', '.ds-store']
```

Number of times a tag appeared 

```python
# https://stackoverflow.com/questions/15115765/how-to-access-sparse-matrix-elements
#Lets now store the document term matrix in a dictionary.

#basically what freqs gets is the number of times a tag is used in whole data .A1 is used to flatten ndarray in general
freqs = tag_dtm.sum(axis=0).A1
result = dict(zip(tags, freqs)
```

```python
#Saving this dictionary to csv files.
if not os.path.isfile('tag_counts_dict_dtm.csv'):
    with open('tag_counts_dict_dtm.csv', 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in result.items():
            writer.writerow([key, value])
tag_df = pd.read_csv("tag_counts_dict_dtm.csv", names=['Tags', 'Counts'])
tag_df.head()

Tags	            Counts
0	.a	             18
1	.app	         37
2	.asp.net-mvc	 1
3	.aspxauth	     21
4	.bash-profile	138
```

Sorting all tags on basis of its freq in descending order 

```python
tag_df_sorted = tag_df.sort_values(['Counts'], ascending=False)
tag_counts = tag_df_sorted['Counts'].values
```
```python
plt.plot(tag_counts)
plt.title("Distribution of number of times tag appeared questions")
plt.grid()
plt.xlabel("Tag number")
plt.ylabel("Number of times tag appeared")
plt.show()
```
![Screen Shot 2021-10-28 at 6 34 17 AM](https://user-images.githubusercontent.com/90976062/139168700-fb00d179-24e6-429d-ba55-a710370b4ea1.png)
If we see the plot its highly skewed falls very rapidly and not much we can get from such plot  so lets zoom in little lets take first 10000 tags only 

```python
plt.plot(tag_counts[0:10000])
plt.title('first 10k tags: Distribution of number of times tag appeared questions')
plt.grid()
plt.xlabel("Tag number")
plt.ylabel("Number of times tag appeared")
plt.show()
```
![Screen Shot 2021-10-28 at 6 37 43 AM](https://user-images.githubusercontent.com/90976062/139168935-005ff297-4443-4fb9-ad57-94c866e747b1.png)

```python
plt.plot(tag_counts[0:500])
plt.title('first 500 tags: Distribution of number of times tag appeared questions')
plt.grid()
plt.xlabel("Tag number")
plt.ylabel("Number of times tag appeared")
plt.show()
print(len(tag_counts[0:500:5]), tag_counts[0:500:5])
```
![Screen Shot 2021-10-28 at 6 44 22 AM](https://user-images.githubusercontent.com/90976062/139169463-7f1b39fa-573c-4a45-afb0-0ce98243bc24.png)

we can conclude one thing that freq of occurence of tags fall very sharply bunch of tags having very high frq followed by very low occuring tags 
```python
# Store tags greater than 10K in one list
lst_tags_gt_10k = tag_df[tag_df.Counts>10000].Tags
#Print the length of the list
print ('{} Tags are used more than 10000 times'.format(len(lst_tags_gt_10k)))
# Store tags greater than 100K in one list
lst_tags_gt_100k = tag_df[tag_df.Counts>100000].Tags
#Print the length of the list.
print ('{} Tags are used more than 100000 times'.format(len(lst_tags_gt_100k)))

153 Tags are used more than 10000 times
14 Tags are used more than 100000 times
```
Observations:

There are total 153 tags which are used more than 10000 times.

14 tags are used more than 100000 times.

Most frequent tag (i.e. c#) is used 331505 times.

Since some tags occur much more frequenctly than others, Micro-averaged F1-score is the appropriate metric for this probelm.

Tags Per Question

```python
#Storing the count of tag in each question in list 'tag_count' as axis = 1 so counts no of tags in a question
tag_quest_count = tag_dtm.sum(axis=1).tolist()

#Converting list of lists into single list, we will get [[3], [4], [2], [2], [3]] and we are converting this to [3, 4, 2, 2, 3]
#https://thispointer.com/python-convert-list-of-lists-or-nested-list-to-flat-list/
tag_quest_count=[int(j) for i in tag_quest_count for j in i]
print ('We have total {} datapoints.'.format(len(tag_quest_count)))

print(tag_quest_count[:5])

We have total 4206314 datapoints.
[3, 4, 2, 2, 3]
```
```python
print( "Maximum number of tags per question: %d"%max(tag_quest_count))
print( "Minimum number of tags per question: %d"%min(tag_quest_count))
print( "Avg. number of tags per question: %f"% ((sum(tag_quest_count)*1.0)/len(tag_quest_count)))

Maximum number of tags per question: 5
Minimum number of tags per question: 1
Avg. number of tags per question: 2.899440
```
```python
sns.countplot(tag_quest_count, palette='gist_rainbow')
plt.title("Number of tags in the questions ")
plt.xlabel("Number of Tags")
plt.ylabel("Number of questions")
plt.show()

Observations:
Maximum number of tags per question: 5
Minimum number of tags per question: 1
Avg. number of tags per question: 2.899
Most of the questions are having 2 or 3 tags
```
![Screen Shot 2021-10-28 at 7 13 27 AM](https://user-images.githubusercontent.com/90976062/139171841-106779da-5fee-492f-a4c9-f85ce065e744.png)

The top 20 tags

```python
i=np.arange(30)
tag_df_sorted.head(30).plot(kind='bar')
plt.title('Frequency of top 20 tags')
plt.xticks(i, tag_df_sorted['Tags'])
plt.xlabel('Tags')
plt.ylabel('Counts')
plt.show()
```
![Screen Shot 2021-10-28 at 7 27 03 AM](https://user-images.githubusercontent.com/90976062/139172978-5da2dcba-61b8-4243-b2b8-61741ee8bb6e.png)

Observations:

Majority of the most frequent tags are programming language.

C# is the top most frequent programming language.

Android, IOS, Linux and windows are among the top most frequent operating systems.

Cleaning and preprocessing of Questions

Preprocessing :

Sample 1M data points.

Separate out code-snippets from Body.

Remove Spcial characters from Question title and description (not in code).

Remove stop words (Except 'C').

Remove HTML Tags.

Convert all the characters into small letters.

Use SnowballStemmer to stem the words.

```python
def striphtml(data):
    cleanr = re.compile('<.*?>')#combines all the mention characters together in cleanr 
    cleantext = re.sub(cleanr, ' ', str(data))# replaces all of them with space for a given data that is converted to str
    return cleantext
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")
```

```python
#http://www.sqlitetutorial.net/sqlite-python/create-tables/
# creating a new conection to sqlite 
def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by db_file
    :param db_file: database file
    :return: Connection object or None
    """
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)
 
    return None
#function to create a new table 
def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)
 
 #function to check if the table exists or not
def checkTableExists(dbcon):
    cursr = dbcon.cursor()
    str = "select name from sqlite_master where type='table'"
    table_names = cursr.execute(str)
    print("Tables in the databse:")
    tables =table_names.fetchall() 
    print(tables[0][0])
    return(len(tables))
    
#creating new database that will contain our preprocessed data     
def create_database_table(database, query):
    conn = create_connection(database)
    if conn is not None:
        create_table(conn, query)
        checkTableExists(conn)
    else:
        print("Error! cannot create the database connection.")
    conn.close()

sql_create_table = """CREATE TABLE IF NOT EXISTS QuestionsProcessed (question text NOT NULL, code text, tags text, words_pre integer, words_post integer, is_code integer);"""
create_database_table("Processed.db", sql_create_table)
