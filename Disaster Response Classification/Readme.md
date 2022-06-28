**Disaster Response**
-----------------------
Disaster response organizations receive millions of messages following a disaster and because different organizations take care of different parts of the problem, there needs to be a way of directing messages to the appropriate organization so that they can respond to the problem accordingly. This web application was built to classify disaster messages so that an emergency professional would know which organization to send the message to. The application uses a classifier that was trained on the data described below. This is part of the Udacity Data Science Nanodegree and this is only a basic version of the application intended for learning.

**Data**
-----------------------
The dataset contains 26,386 labeled messages that were sent during disasters around the world. Each message is labeled as 1 or more of the following 36 categories:

_'related', 'request', 'offer', 'aid_related', 'medical_help', 'medical_products', 'search_and_rescue', 'security', 'military', 'child_alone', 'water', 'food', 'shelter', 'clothing', 'money', 'missing_people', 'refugees', 'death', 'other_aid', 'infrastructure_related', 'transport', 'buildings', 'electricity', 'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure', 'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold', 
'other_weather', 'direct_report'
_

None of the messages in the dataset were labeled as child_alone so this category was removed altogether before building the classifier, leaving 35 categories to classify.

**Classifier Selection**
-----------------------

**Pipeline**

To classify these 35 categories, this application uses a machine learning pipeline with the following steps:

* Tf-idf vectorizer - tokenizes an entire corpus of text data to build a vocabulary and converts individual documents into a numeric vector based on the vocabulary

* Tokenizer steps: lowercase all characters > remove all punctuation > tokenize text into individual words > strip any white space surrounding words > remove stopwords > stem remaining words

* Vectorizer steps: convert a text document into a term frequency vector (word counts) > normalize word counts by multiplying the inverse document frequency

* Multi-output classifier using 3 different algorithms: Random Forest, Naive Bayes and ADA Boost - predicts 35 binary labels (0 or 1 for each of the 35 categories)

* Metrics results were compared for each and the best pipeline was selected based on Accuracy, Precision and Recall values for the ML pipeline script

* The classifiers were also tested with some text messages to see how they predicted the categories

**Results**

The ADA Boost classifier showed the highest values in terms of the Accuracy metric. It was evaluated on a test dataset with the following results:

- accuracy                  0.948286
- macro_avg_precision       0.801429
- macro_avg_recall          0.636286
- macro_avg_f1              0.668000
- weighted_avg_precision    0.940857
- weighted_avg_recall       0.948286
- weighted_avg_f1           0.938286

**Files**
-------------------

**ETL Pipeline**

File data/process_data.py contains data cleaning pipeline that:

* Loads the messages and categories dataset
* Merges the two datasets
* Cleans the data
* Stores it in a SQLite database

**ML Pipeline**
File models/train_classifier.py contains machine learning pipeline that:

* Loads data from the SQLite database
* Splits the data into training and testing sets
* Builds a text processing and machine learning pipeline
* Trains and tunes a model using GridSearchCV
* Outputs result on the test set
* Exports the final model as a pickle file

**Flask Web App**
Run the following commands in the project's root directory to set up your database and model.

To run ETL pipeline that cleans data and stores in database python: data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl

Run the following command in the app's directory to run your web app python run.py

Go to http://0.0.0.0:3000/

**Plots visualized**

![newplot (1)](https://user-images.githubusercontent.com/28513435/172015450-972279f3-9498-49c6-be4c-5768268855bc.png)


![newplot (2)](https://user-images.githubusercontent.com/28513435/172015499-9b73f493-4ccf-4d57-b8c6-d00e8d506e7d.png)


![newplot (3)](https://user-images.githubusercontent.com/28513435/172015520-1e23faeb-8302-4999-bdec-da50fbf70834.png)


**MOST IMPORTANT: Improvements that can be made**
--------------------------------------------------
1. Improve the ML pipeline to check other classification algorithms and identify the most accurate one
2. Improve the design of the web application and plot more visualizations for better insights into the data
3. Learn and improve on the NLP algorithms to see how better the messages can be classified to ensure in a real-life scenario, the correct and relevant help can be dispatched based on the message.
