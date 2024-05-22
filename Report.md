# Unsupervised Learning-Based Comparison for News Article Topic Classification

## Duration
Dec 17, 2023 ~ Jan 15, 2024

## Abstract
In my university coursework, I embarked on implementing k-means clustering and an Artificial Neural Network (ANN) model using data I collected myself. During the process, I learned about a newly released model called Gemini on Google. Intrigued, I decided to compare k-means, Gemini, and GPT, all of which I had learned during class, using my collected data.

Surprisingly, the Gemini model exhibited higher accuracy compared to k-means and GPT. This project report aims to provide insights into the performance of these three models in the context of text analysis, shedding light on the advancements brought by Gemini.

## Introduction
### Why I Built a Classification Model with Unsupervised Learning
The ability to classify things is a core skill in AI. Every new AI model has it, and as an AI student, I wanted to build one myself. But instead of the usual supervised learning, I used a different approach: **unsupervised learning**. This means the model learns on its own, without needing pre-labeled data.

For this project, I compared three different models:
1.  **k-Means Clustering:** This is a classic algorithm from the Scikit-learn library. I used it to group similar data points and visualize the results with Matplotlib.
2. **Gemini:** This is a brand new Large Language Model from Google.
3. **GPT from OpenAI:** This model is older than Gemini, but it's still powerful. 

I was curious to see how these different models would perform with unsupervised learning. I hope this project shows my understanding of AI concepts and my ability to apply them to real-world problems.

## Material & Method
### 1. Collect text data
I gathered 1,000 news articles from November 22, 2023, to December 12, 2023, using the 'News API.' To maintain political balance, I selected three different news sources: The Chicago Tribune (progressive), The Wall Street Journal (conservative), and Reuters (centrist).

### 2. Data Normalization
I used the NLTK module for the tokenization and normalization of the dataset. Word vector embedding was done using the Word2Vec module. Titles, descriptions, and authors were processed separately and compared.

### 3. Train text classification model
Titles, descriptions, and authors were modeled separately and compared using three methods.

#### 3.1 k-Means clustering
I applied NLTK for tokenization and normalization. Word2Vec provided word vector embeddings, and I used the average sentence vector for training, incorporating TF-IDF values as weights.
#### 3.2 GPT
Could does not need to be normalized in the data set, but, it needs the data set in JSON format. While converting to JSON format, the length of the data set increases by 1,000 to 90,978. Therefore, normalization of the data set before training the model.

#### 3.3 Gemini
Normalization of the data set Does not need to be done before training the model. 

## Result
### 1. k-Means Clustering
There are five clusters, Politics, Society, World, Environment, and Sports. 
Compare four different combinations using the Silhouette Score.
- Everything (Title, Description, Author): 0.23532203200038282
- Title and Description: 0.33557408077411455
- Title and Author: 0.315865400824296
- Description and Author: 0.3153713939008773

Due to the higher value of the silhouette score, use the Title and Description combination for k-Means clustering. 

![Cluster visualization](/img/cluster_visualization.png)

#### The result of five of each cluster 
- Cluster 0 (164 articles) - Mixed (Environment, Politics, Society) 
	0 Real Estate Matters: Heir seeks clarity on possible taxes owed for sale of inherited property
	2 Explainer: Global fossil fuel subsidies on the rise despite calls for phase-out
	12 Man, 30, seriously wounded overnight in shooting on Far South Side, police say
	13 Thanksgiving forecast: Sunny conditions expected with highs in mid 40s
	22 2024 E-Ray: The Quickest 'Vette Ever...

- Cluster 1 (207) - Mixed (Politics, Society, Sports): 
	1 Three men charged following Midlothian liquor store robbery
	3 Police investigating 3-hour Thanksgiving armed robbery spree on West, Southwest Sides
	5 Jordan Love ties career high with 3 TD passes, leads Packers to 29-22 win over NFC North-leading Lions
	6 Nikki Haley’s Medicare Advantage
	16 Retailers are ready to kick off Black Friday just as shoppers pull back on spending

- Cluster 2 (123) - Mixed (Sports, Society)
	7 Chicago Blackhawks’ Taylor Hall is expected to miss the rest of the season with a right knee injury
	8 The 89th annual Chicago Thanksgiving Parade
	10 Norovirus outbreak investigation linked to $1 burrito event underway in Evanston, health department confirms
	11 Exclusive: Barclays working on $1.25 billion cost plan, could cut up to 2,000 jobs -source
	17 2 fatally wounded in separate shootings Wednesday evening on South Side, Chicago police said

- Cluster 3 (284) - Mixed (Environment, Economics, Sports)
	15 To save the climate, the oil and gas sector must slash planet-warming operations, report says
	20 Return to Work Coming for Your Pandemic-Era Home...
	21 Bankman-Fried's Life Behind Bars: Crypto Tips and Paying With Fish...
	24 What’s J.J. McCarthy’s secret to staying focused? Daily meditation, says the Michigan QB and Nazareth alumnus.
	30 Editorial: Our thanks to Chicago firefighters and their families

- Cluster 4 (222) - Mixed (Society, World)
	4 Giant floats return, drawing thousands to 89th Thanksgiving Parade in Loop
	9 Cops warn businesses of overnight burglaries on 47th Street in Kenwood neighborhood
	14 What does Sam Altman’s firing — and quick reinstatement — mean for the future of AI?
	19 Don't press 'pandemic panic button' scientists caution on China pneumonia report
	31 Soot pollution from coal-fired power plants is more deadly than soot from other sources, study shows

#### Result of ANN classification
After training the model with the training set and evaluate with the test set, the accuracy is 0.9750000238418579.

### 2. GPT
There are five clusters, Politics, Society, World, Environment, and Sports. 
Compare four different combinations using the Silhouette Score.
-  Everything (Title, Description, Author) : 0.25316664576530457
- Title and Description : 0.32652321457862854
- Title and Author : 0.29335862398147583
- Description and Author : 0.2955780029296875

Due to the higher value of the silhouette score, use the Title and Description combination for k-Means clustering. 

![GPT k-means visualization](/img/GPT_kmeans_visualization.png)

#### The result of five of each cluster 
- Cluster 0 Articles (167) - Mixed (Politic, Society, Environment):
	4     Giant floats return, drawing thousands to 89th Thanksgiving Parade in Loop
	6      Nikki Haley’s Medicare Advantage
	13    Thanksgiving forecast: Sunny conditions expected with highs in mid 40s
	14    What does Sam Altman’s firing — and quick reinstatement — mean for the future of AI?
	16    To save the climate, the oil and gas sector must slash planet-warming operations, report says - **misclassified**

- Cluster 1 Articles (219) - Environment :
	2     Explainer: Global fossil fuel subsidies on the rise despite calls for phase-out
	10    Norovirus outbreak investigation linked to $1 burrito event underway in Evanston, health department confirms
	11    Exclusive: Barclays working on $1.25 billion cost plan, could cut up to 2,000 jobs -source
	15    To save the climate, the oil and gas sector must slash planet-warming operations, report says
	19    Don't press 'pandemic panic button' scientists caution on China pneumonia report

- Cluster 2 Articles (263) - Sports:
	5     Jordan Love ties career high with 3 TD passes, leads Packers to 29-22 win over NFC North-leading Lions
	7     Chicago Blackhawks’ Taylor Hall is expected to miss the rest of the season with a right knee injury
	8     The 89th annual Chicago Thanksgiving Parade
	18    4 takeaways from the Chicago Blackhawks’ 7-3 loss, including Bedard vs. Fantilli Part I and the ‘A-Teens’ power play debut
	24    What’s J.J. McCarthy’s secret to staying focused? Daily meditation, says the Michigan QB and Nazareth alumnus.

- Cluster 3 Articles (148) - Mixed (Society, Economic):
	0     Real Estate Matters: Heir seeks clarity on possible taxes owed for sale of inherited property
	20    Return to Work Coming for Your Pandemic-Era Home...
	45    Stellantis recalls more than 32,000 hybrid Jeep Wrangler SUVs
	46    Track coach pleads guilty tricking women into sending him nude photos
	47    U.S. egg producers conspired to fix prices, Illinois jury finds

- Cluster 4 Articles (203) - Society :
	1     Three men charged following Midlothian liquor store robbery
	3     Police investigating 3-hour Thanksgiving armed robbery spree on West, Southwest Sides
	9     Cops warn businesses of overnight burglaries on 47th Street in Kenwood neighborhood
	12    Man, 30, seriously wounded overnight in shooting on Far South Side, police say
	17    2 fatally wounded in separate shootings Wednesday evening on South Side, Chicago police said

### 3. Gemini
There are five clusters, Politics, Society, World, Environment, and Sports. 
Compare four different combinations using the Silhouette Score.
 - Everything (Title, Description, Author) : 0.24500106275081635
- Title and Description : 0.31558099389076233
- Title and Author : 0.2562601864337921
- Description and Author: 0.2814783751964569

Due to the higher value of the silhouette score, use the Title and Description combination for k-Means clustering. 

![Gemini k-means visualization](/img/Gemini_kmeans_Visulalization.png)

#### The result of five of each cluster 
- Cluster 0 Articles (165) - Society :
	4     Giant floats return, drawing thousands to 89th Thanksgiving Parade in Loop
	27    Editorial: Abe Lincoln‘s generous gift: A day for Americans to give thanks—and eat turkey
	28    Letters: Our nation would be wise to be thankful for its immigrants
	34    We Should All Give Thanks for Taylor Swift
	36    Grateful for a new home: Community members embrace migrants as many find meaning in their first Thanksgiving celebration in Chicago

- Cluster 1 Articles (221) - Mixed (Environment, Economics):
	0     Real Estate Matters: Heir seeks clarity on possible taxes owed for sale of inherited property
	2     Explainer: Global fossil fuel subsidies on the rise despite calls for phase-out
	10    Norovirus outbreak investigation linked to $1 burrito event underway in Evanston, health department confirms
	11    Exclusive: Barclays working on $1.25 billion cost plan, could cut up to 2,000 jobs -source
	15    To save the climate, the oil and gas sector must slash planet-warming operations, report says

- Cluster 2 Articles (220) - Mixed (Sports, Society):
	5     Jordan Love ties career high with 3 TD passes, leads Packers to 29-22 win over NFC North-leading Lions
	7     Chicago Blackhawks’ Taylor Hall is expected to miss the rest of the season with a right knee injury
	8      The 89th annual Chicago Thanksgiving Parade
	18   4 takeaways from the Chicago Blackhawks’ 7-3 loss, including Bedard vs. Fantilli Part I and the ‘A-Teens’ power play debut
	24    What’s J.J. McCarthy’s secret to staying focused? Daily meditation, says the Michigan QB and Nazareth alumnus.

- Cluster 3 Articles (203) - Politics:
	6     Nikki Haley’s Medicare Advantage
	14    What does Sam Altman’s firing — and quick reinstatement — mean for the future of AI?
	16    Retailers are ready to kick off Black Friday just as shoppers pull back on spending
	20    Return to Work Coming for Your Pandemic-Era Home...
	29    David McGrath: I give thanks for these fine Americans

- Cluster 4 Articles (191) - Society:
	1     Three men charged following Midlothian liquor store robbery
	3     Police investigating 3-hour Thanksgiving armed robbery spree on West, Southwest Sides
	9     Cops warn businesses of overnight burglaries on 47th Street in Kenwood neighborhood
	12   Man, 30, seriously wounded overnight in shooting on Far South Side, police say
	13    Thanksgiving forecast: Sunny conditions expected with highs in mid 40s

#### Result of Gemini classification
Result: loss: 0.5434 - accuracy: 0.8100 {'loss': 0.5433962345123291, 'accuracy': 0.8100000023841858}
![Gemini Classifier Performance](/img/Gemini_Classifier_Performance.png)

## Conclusion
Compared to GPT and Gemini, Gemini is more user-friendly. Gemini is easy to use and there are up-to-date guides. OpenAI has a guide for using it, but it's outdated and requires some preprocessing. The result of the clustered data is more accurate when using the Gemini model than k-means clustering.
These training models were unsupervised learning; this makes the accuracy of models lower. 
Additionally done ANN classification and Gemini classification. Also, Gemini shows more accuracy than the ANN model.
