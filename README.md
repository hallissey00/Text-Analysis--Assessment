# Text-Analysis--Assignment

 Mine non-structured (text) data using Python libraries.
 
  Overview
The three categories of the topic that will be related to the project is Xbox, PlayStation and the Premier League football. The two topic that are very similar they are two gaming consoles and the other topic is based on football the premier league. 

Business objective
The business objective of this project is to find the dataset of the three topics I choose . The three datasets will contain attributes and an class label. There are general attributes such as articles which will come up all the articles related to the topics that I’ve chosen. Which are PlayStation, Xbox and the premiere league. The plan for this project is to get the main article for the three main topics and have the articles print out for the classes. 

Mining objective
This stage should generate a model to see how many words are in a class. The mining objective will be process collections of text with document which PlayStation, xbox and  premiere League.  Text Mining objective is a multi-stage process. With some informational of the content included in any of the items like newspaper, articles, books, reports,  blogs and articles. The goal of the mining objective is to identify the useful information without duplication from various documents with synonymous understanding and this is what we be doing related to the mining objective

The process employed 
There are 30 articles/documents with each class which be for premiere league and two classes that should be similar. Which is xbox and playstation 
The goal is to focus on related textual data which shows text that can be stored in very large datasets. We will use the  Beautiful Soup which is a library which makes it easy for us to scrape any information from web pages. To do this we will have to use a web structure mining tool, it would make us find a lot of useful information and knowledge of an analysis hyperlinks. With the 90 source pages that we have some of the links would have to be removed for example maybe they didn’t have a image/picture and wasn’t very structured to be able to add the link to the 90 sources. The scraping process has two main libraries which be requests and bs4. Request library will used to send are HTTP link request to a website and then it will store our response object within a variable, how we fetch the request is by  using a beautiful soup which will catch the data using Html tag, class, etc.  The second main library is the BS4 which will be used for pulling the data out of the HTML and XML files. 

Select the relevant html elements data,
The text content of the relevant html elements, The category’s for premier league , xbox and playstation,  all each have a URLs to retrieve data from the h1 and p-elements, we can use the len for printing the length for documentation. Each class will contain the length of 30 documents. When all classes are together we use all_docs. This is used by displaying  a result for all documents which be 90. We get text by using h1 and p. H1 and P are important elements to the articles. Why we use the developer tools is for the structure of the websites and clarify every elements that are important. The main developer tools that be scraped is h1 and p.  

Building the corpus 
Before the corpus be built we research more related to corpus to see how it works. The corpus is collection of documents, books or articles. It can be split into individual documents and also categories. It then be broken down into sentences, words or characters. We have to import the NLTK, this can offer some functionalities for corpus reading. We can use an Domain Specific Corpora tool for building application related to a specific language models. After we build a Domain Specific Corpora, we then map each document to a corresponding word from using a dictionary structure. After that’s all done we use dataframe structures to build a labelled corpus, by using build corpus an “def build_corpus(docs,labels)” for literally  building the corpus, then after use “corpus = build_corpus(all_docs, all_labels)” to relate to all of the 90 source documents. Once the corpus is built, it be written as a csv file, then we use the csv function from importing pandas.
Thw 3 classes and the labels are then finalled contained into one. The corpus DataFrame contains the article and class with there documents, therefore its why we use len for the corpus. The corpus display random rows and columns example  90 rows and columns. With the corpus file called corpus.cvs already we use this  by reduce the size of each of the document. 

Baseline Models 
Derive 3 matrices using 3 vectorisation techniques
The three matrices which be able to use for the baseline model is counts, normalised counts and tfidf. The matrices is important mainly because we can extract set of unique words/tokens from the documents and we can use them to create an matrix with entries that will let us know what occurred in the document. The matrices generate every document that’s in our dataset which are dataset be for our project is corpus dataset.  Tackling this task we will use three matrices. Start of using first of use def build_count_matrix to tokenise the corpus, then we will create group list of dictionaries they be are frequency distributions. while documenting tokenised in the def build count we generate a dictionary in the documents. 
We start of now using a def build_tf_matrix method. We use this function to generate matrix with normalised frequencies. The build_tf_matrix uses the count_matrix and the doc_lengths for normalising the frequency.  The third matrices uses the def build_tfidf_matrix(docs). Therefore the matrices function will generate a matrix with tfidfs scores. 
Conclusion of this this be the tf is will have the number of times that a word occurs in the documents but will divide by total number with words in the document however a cell entry represents normalized frequencies count of total number of words in a document. for the Derive 3 matrices using 3 vectorisation techniques is the difference between the three matrices and the count matrices will create list of dictionaries and generate an count for dictionary in each document. The Idf  is number documents which be collection that is divide by number documents where words occurs and the normalised matrices will generate the matrices for the normalised frequencies and the tfidf matrices will generate a score. The resulting matrix contains features of 12443 columns attributes with 90 documents.

 

Choose at least 1 classification algorithm for baseline modelling
The 1 classification algorithm to use with the baseline model will be the Decision tree. We used  the decision tree analysis for outlining outcomes, costs, and frequencies of decision that’s complex. Why we use this classification is because its easy analysing quantitative data and having a decision that’s based with numbers and is very helpful finding result. This will make it all easy to use and apply it to the 3 matrices and td our result. Another option was KNeighborsClassifier but thought the decision tree be better to use. 

Apply the algorithm to the 3 matrices 
We used the decision tree algorithm to the 3 matrices by using the crossvalidate_model for their performances. By apply with the decision tree matrices we use “DecisionTreeClassifier(random_state=1)” to randomly state the scores, then finding the data class with using “y = data.Class”. 
We will then use a cross validation method for finding performance with using “crossvalidate_model(dt_clf, baseline_count_matrix, y, print_=True)”. 
In the crossvalidate_model we put it in our matrices in the brackets of that method and print out the results. For each matrices we use with using the algorithm it be document the results for the image below for example of the result for each algorithm to the 3 matrices. We used to the count. tf and tfidf and find the results for each score time, percission and recall. 

 
Data Understanding 
Derive word/token statistics 
A token for our project would be a unit that represents any word or character sequence as it appears in text. For the project we have 30 token types appears twice. In the project it represents as one type and It’s how Token works meaning we would have to derive a token in a category. 
while using Word Tokenization it divides a text into words that were delimited by characters that act as word boundaries. Word Tokenization is used cause its known for generating word vectors for mining. While finding the number of token statics of 30 of each category, we would use the most Frequequent method to find the text of each token: Premier League : “tm.print_n_mostFrequent("premierleague", premierleague_text, 30)” playstation: tm.print_n_mostFrequent("playstation", playstation_text, 30) xbox: tm.print_n_mostFrequent("xbox", xbox_text, 30). Therefore now we will use the 
attributes =sorted(set(list(baseline_count_matrix.columns))) print(attributes)
to print out the word and token.


visualisations techniques 

Bar charts 
The normalise count for pos with each document takes a list of tagged docs. When we plot the frequency for pos text w ehave it in a list meaning the function would plot a bar charts with Frequency distribution of CC , label Frequency. 
To visualise a frequency for CC part of speech we have to start of with documents to store a text = [premierleague_text, playstation_text, xbox_text], then visualise the frequency of CC part of speech which be conjunctions. Then across the 3 categories we use plot_POS_freq(texts, 'CC', ['premierleague', 'playstation', 'xbox']). 
For the CC frequency the PlayStation is the most occurring at 0.030, but the premier league and xbox are on the same frequency with 0.028.
We then change CC to CD (Cardinal number), the one that oocurs with the most articles or classes is premier league with 0.06, the xbox with 0.03 and the PlayStation this time is the lowest with 0.01.

Frequency distribution of CC
The part of speech for CC, the highest occurring from the documents is Playstation and then xbox and premier league is the same.


Frequency distribution of CD 
The part of speech for CD, the highest occurring from the documents is premier league, then its xbox and last playstation.  


Identify frequently occurring terms, potential stop words, synonyms, concepts, and word variations
We will see what is the most 5 frequently occurring terms, potential stop words, synonyms, concepts, and word variations for each class. We check for each premier league, PlayStation and xbox for the most frequently that occurs.
 

For the premier league for the most 5 frequency tokens: 
[('the', 2924), (',', 1868), ('.', 1505), ('of', 1099), ('in', 956).
‘The’ occurs the most 2924 times and frequency of 0.06, then ','    stop word occurs 1868 times frequency of 0.04


For the xbox frequency for the most 5 frequency tokens :
[(',', 1544), ('the', 1191), ('.', 927), ('to', 868), ('and', 754)]
‘The’ is the most word appeared 1191 times, frequency of 0.03 and the most stop       words occur was ‘,’ 1544 times with frequency of 0.05.

For the xbox frequency for the most 5 frequency tokens :
[(',', 1791), ('the', 1357), ('.', 1185), ('to', 1099), ('a', 963)]
‘The’ is the most word appeared 1357 times, frequency of 0.02 and the most stop      words occur was ‘,’ 1791 times with frequency of 0.03.


Word cloud 
The word cloud is to visual show frequency for terms with each documents for premier league, xbox and playstation.  The word cloud great tool for mainly identifying its frequent/predictive terms. its great for identify synonyms and words variations and it also identify which terms occur for each topics.  I had to use a website to get the word cloud, by trying to use it on jupyter lab it wouldn’t work therefore I used a website that a lecturer provided the class to use. The Website is used to the word cloud for me, the website is wordcloud.com http://www.wordclouds.com. To get the word cloud working we use the corpus.csv to print out each class for the premier league, xbox and playstation. Therefore I was able to get the results for the word cloud to show what words occur the most in the document for the articles.

Premier League 
This is the word cloud is for the premier league. generate the word cloud all the text with the links that’s connected to the premier league. The visual representation with all frequency terms you can see appear the most be club names , season, title, top, season, promoted all these words you see are words that occur the most in our documents for this topic. 

                         
Xbox
This is the word cloud is for the xbox, it generate the word cloud all the text with the links that’s connected to the xbox. The visual representation with all frequency terms you can see appear the most be games, game pass, days, gaming, console and xbox one all these words you see are words that occur the most in our documents for this topic. 
                                   

PlayStation 
This is the word cloud is for the playstation, it generate the word cloud all the text with the links that’s connected to the playstation. The visual representation with all frequency terms you can see appear the most be game, players, lead, new, comments and content all these words you see are words that occur the most in our documents for this topic. 
                                          
Use 2 clustering algorithms with 2 different linkage schemes
The two linkage we will be using is single and complete. We start of explaining single linkage and then the complete linkage for each clustering and linkage schemes. The Agglomerative and K mean clustering.  the cosine is a Asymmetric, it cosine similarity is measured cosine of the angle between vectors. 
Cosine is main cluster for the project mainly cause we now have a means to calculate distance. The first clustering we will use is the K means, we using linkage of single and complete so we will use that first for the kmeans method then we try the agglomerative.

KMeans
The algorithm organises objects into the specified number of clusters. The K means cluster is the objects from the same cluster being to close eachother, and then objects from different clusters are far away from eachother. 
We have rows of the data to be selected at random, they represent a cluster mean and centre. a new mean is calculated for each cluster when the objects are all allocated. Then tuples in the database will be compared to an new mean. That’s how we use the K Mean cluster for the project. 
Single:
We tried the KM_random. Labels for the single to get the result. The is loads of clustering in this. 
 


Completed:
It has less clusterin 1s than it did before there is now more 0s thans 1st while its completed. 
 

Agglomerative Clustering
The Agglomerative Clustering we got to use to connect with the linkage are also single and complete. By getting the Agglomerative to work we got to use the homogeneity and the completeness score to work, they score for an single approach. The only issue was the dendrogram it was to close to each other and couldn’t see the bottom part of the numbers. But overall the algorithms, schemes and measures worked. The affinity performed for this was cosine.

Single: 
You can see we have 1 in one document and 2s which means cluster 1 document
 

Completed:
You can see its been completed and there are now more 2s. so this performed bad.

 
Main Data Preparation
Text preprocessing tasks
The pre-processing tasks main processing for data is initial clean data, improving bag of words and stop words. It can clean data which would reduce number of things that aren’t useful. such. Improving bag of words will reduced number things like the terms that replaced by synonyms. Like replacing word variation game_play to gameplay which is the same. Its changing  hyponyms to corresponding hypernyms. Then Stop words removal removes attributes because mainly lot of filters contain barely any information, with removing them would increase performance big time. 

Initial Cleaning
The first task I am picking for the ask is Initial Cleaning. Initial cleaning it will generate new modules and it counts my projects, but I do this mainly cause it improves scores with fewer attributes. I use this with a count, tfidf and count.
The performance of initial clean data for the count has test accuracy 1.00, the test precision macro is 1 and same as test recall and then the standard deviation is 0
for the tf matrix and tfidf both have same with test accuracy 0.97, the test precision and test recall macro is 0.97 but different standard deviation. 
Therefore the matrix count has a higher test accuracy, test precision and test recall.

 
Improving Bag of Words
The second task is Improving Bag of Words which can improve the project including recognising word relations, Recognise informal word and Recognise Collocations of phrases or expressions and Linking canonical forms. 
We use these to improve the bow for our project. But we replace synonyms we use    'premierleague': ['premier[_]?league'],
  'xbox': ['gaming'],
  'playstation': ['\bgame(s)?\b', 'player(s)?', 'platform(s)?']
We do concepts to replace general  selection to identify. The results we with the number of terms after improving the bow was very high it was 16715.

 

Stop words Removal
The third task was Stop words Removal. I used this to reduce the number features, it  be based on the text exploration. We use the custom list for stop removal words, meaning it decreases all scores and we will stick with just using the universal one svm for the project, mainly cause svm one that works perfect for my project.
The terms after removing universal sw is 12109.

 
 
Algorithms-based Feature selection/reduction tasks
I will now have to choose least 2 techniques for the algoritm based feature selection to reduce the tasks. I will Document and discuss the performance after each applied technique and decide which one to include in the final pipeline. The two technique I will test out is meta section and RFE selection.

Meta Section 
The meta section shows amount times word occurs in documents. for example. The meta section performances is high, the accuracy, precision and recall is both 96% and terms applying anova is high aswell with 2874.

 

RFE selections 
The RFE selections is to get rid of features by selecting number of features to eliminate. RFE seem to be higher than the meta section. Its accuracy, precision and recall is 98% which is very high even higher than 96% meta. Therefore we are going to use the RFE selection for the final pipeline. 
 


Build Classification Models
Choose another suitable algorithm
The other suitable algorithm to use for baseline matrices is KNeighborsClassifier we use this algorithm to output 3 matrices. I choose KNeighbors algorithm for the baseline modelling mainly cause it could also give good results.
The algorithm I choose at the start was the decision tree. I will compare the both of the algorithms.
The count matrices accuracy for KNeighbours, precision and recall has percentage of 99% and same goes with the decision tree. With the Kneighbours having 98% with the tfidf and the tf that is exactly same percentage as the decision tree. By looking at the score time and the standard deviation seems like the KNeighbours and the Decision tree algorithms are exactly the same with the three matrices. 
 


Parameter tuning
The algorithm that I picked for the parameter turning is DT , tunes parameters with criterion max_depth , min_samples_split, example:
•	min_impurity_decrease is value of threshold with split internal nod.
•	max_depth is the maximum number of nodes which grow decision till its cut off.
•	min_samples_split is minimum number of rows split internal node.
•	min_samples_leaf the minimum numbers, requires to be present in the leaf node. 
The hyperparameter tuning will have criterion shown that is choose an gini option.        The max_depth is 3.  The min_samples_leaf  is 3 meaning  minimum samples need to present in the leaf, has 3% in the data min.  the min_samples_split is 2 mean number rows to split in the internal node, The min_impurity_decrease is 0.01 in the threshold. 

Hyperparameter:
 
The Optimal parameters stat is random for all the parameters from hyperparameter, then we can finally find the performance. The Accuracy wend up being very high with 0.98. the precision is 0.98 and recall also 0.98. therefore the are all  0.98


Overall evaluation

a)	Which vectorization technique produced best results?
The count technique has performed gave the best result with 98%, compared to tf matrix and tfidf matrix. The Tf matrix accuracy is 0.97%, precision and tfidf is 97% aswell. 
Best result = count technique.


b)	Which preparation techniques produced best results?
The initial cleaning had a 97% performed well but the stop removal performed the best with 98% and the lowest was the bags of words with 94%. 
Best result = Stop Removal.


c)	Overall, which classifier produced the best results
The best result for the classifier is the count matrix performed 99% and the tfidf and tf had the same again which was 98%.  The count test_accuracy: 0.99,
test_precision_macro: 0.99 and test_recall_macro: 0.99 all 0.99, with tfidf and tf have around 0.98 or under.
Best Results=count matrix


d)	Overall, which model (pipeline) produced best result (include vectorisation, preparation, and the classifier and its optimal parameters in your answer)
The model with best results vectorisation is the count with 97%.
The best result of preparation for the pipeline is the stop removal 98%.
The optimal parameters best results was the tfidf with 98%.

e)	What features/terms were the most predictive?  
The most predictive features pred terms are league, play and game.


f)	Do the classification results concur the results of the clustering and to what extent? 
The performance of the classification with test accuracy of 0.89%, precision of 0.89% and recall of 0.89%

g)	Discuss the limitations of your project and provide recommendations for future improvements 
The limitation of the project was that there was 30 urls links for each of the three class documents, articles, websites etc but overall was 90 urls. But overall can have better links as improvements, urls were sometimes not working therefore had to get a new url link related to premier league, xbox or playstation.

