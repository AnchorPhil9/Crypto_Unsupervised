# Unsupervised Machine Learning with Cryptocurrency Data
We have been tasked to create a report on cryptocurrencies being traded. For this project, we have obtained cryptocurrency data from CryptoCompare. Once we process all the data for currently traded cryptocurrencies, we will use unsupervised machine learning to group that data together and gain a better view of the cryptocurrency market.

## Data Processing
A quick glance of the starting data shows us three important parameters: trading status, crypto supply, and crypto mining. To prepare the data for unsupervised machine learning, we used astype(int) and other DataFrame functions to store data on currently traded crypto in a new dataframe called **clustered_df**. Given the variety of cryptocurrency prooftypes and algorithms to account for in our analysis, we used get_dummies on the **"Prooftype"** and **"Algorithm"** columns for standard scaling. In turn, we use Principal Compenent Analysis (PCA) to reduce the dimensions of the scaled data to just 3 dimensions, which will make things easier for our machine learning work.

# Data Clustering
With our data of tradeable crypto ready, we could now cluster the data together. First, we ran an elbow curve function to determine our ideal number (4) of KMeans clusters. Then we prepared a KMeans model using that many clusters, training the model with our processed data so that we can predict what clusters each trading cryptocurrency belongs to. After storing those predictions as labels inside a **clustered_df** column, we made a 3-d scatter plot to visualize the clusters. For a 2-d scatter plot, we had to scale the data on crypto supply and crypto mining via MinMaxScaler() then store that data into a new dataframe, which we then used to plot crypto mining against crypto supply.

# Findings
