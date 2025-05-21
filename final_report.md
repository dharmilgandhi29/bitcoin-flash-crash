**Predicting Movement Events in Bitcoin Prices Using Common Metrics and Sentiment Analysis**

**Team: ML-Mavericks**

1. Alex Grimm, Github ID: agrimm02 \[POC\]  
2. Avi Sharma, Github ID: Avish16  
3. Dharmil Milan Gandhi, Github ID: dharmilgandhi29  
4. Hari Ram Selvaraj, Github ID: SNIPOFIST

**Introduction**

Cryptocurrency is becoming an increasingly popular type of investment for people across the world. Cryptocurrency markets are open 24/7 for every day of the year. However, as these markets have become more popular, their prices have experienced more variation. In other words, the prices of these popular cryptocurrencies tend to change very often. Our project aims to predict when flash crashes and other similar market events will occur in certain cryptocurrencies. A flash crash occurs when the price of a stock drops suddenly, but then quickly recovers (Will Kenton, Investopedia.com). Events like these can cause problems, but they also offer some opportunities. In terms of problems, if an investor or company is using some sort of program to do their trading for them, the program may not be able to recognize when a flash crash might occur. If the trading algorithm is not being monitored, the algorithm could sell the stock during this type of event causing the investor to lose money. On the other hand, if an investor wants to purchase a certain cryptocurrency, buying it while the price is down is optimal, so understanding when a flash crash is potentially going to occur is important. 

This research aims to develop a predictive model capable of detecting the occurrence of a flash crash in cryptocurrency markets before it actually happens. Being different from other traditional financial markets, which are well regulated, the market of cryptocurrencies works in a completely decentralized and unregulated way, which may result in extreme oscillations in prices. Our approach will integrate anomaly detection and sentiment analysis for a robust predictive system. The existing methodologies focus on post-event analysis rather than real-time prediction. The project will use machine learning techniques, more specifically an LSTM to enable real-time alerts to traders, exchanges, and regulators with the aim of reducing financial risks from flash crashes. The implications of this study are not limited to traders and investors. These bits of information can be used by regulators in monitoring market manipulation and consequent policy interventions, while cryptocurrency exchanges can refine their risk management frameworks. Algorithmic trading companies can leverage such insights to make necessary adjustments to their trading strategies for periods of high volatility and avoid losses. The success of this project may be a gateway to improved financial stability in cryptocurrency markets and an influencer of the general adoption of predictive modeling in high-volatility assets.

**Literature Review**

Most of the existing literature on flash crashes is related to conventional equity and forex markets. Johnson et al. (2013) demonstrated that an algorithmic trading anomaly, primarily HFT, plays a critical role in triggering such extreme price dynamics. The HFT systems execute trading in milliseconds to amplify the volatility of the market, thus causing flash crashes. Feuerriegel and Gordon (2018) studied the use of anomaly detection approaches, such as Isolation Forests and DBSCAN, in recognizing abnormal patterns in trading and highlighted, through that study, the importance of machine learning in predictive financial modeling.

The studies in cryptocurrency flash crashes are rather few but emerging. Akyildirim et al. (2020) analyzed the 2019 Kraken Bitcoin flash crash. The major factors identified to have driven the crash included low liquidity, order book imbalance, and liquidation cascades. Their findings gave insight into how digital asset markets are so prone to sudden, sharp price distortions and called for predictive frameworks able to provide warning signals in advance. While their study was primarily a post-event analysis, our research extends their findings to incorporate predictive methodologies that can provide warnings in real time.

Recent studies have also broadened the scope of flash crash research toward security risks in DeFi. Oosthoek (2021) conducted an investigation into the cyber threats in DeFi and focused on how adversaries manipulate smart contracts and liquidity pools to create flash crashes for financial gains. These findings underscore the difference between organic market movements and manipulation-induced crashes, thus reinforcing the need for anomaly detection models with real-time monitoring. Wehrli and Sornette 2022 discussed the Hawkes (p, q) framework-a mathematically handy way to divide flash crashes into endogenously and exogenously triggered ones. The research postulates most flash crashes create self-exciting dynamics so that an initial price drop triggers further selling pressure, leading to a larger degree of instability. By integrating such insights, our model seeks to adopt a multifactor approach in order to capture both fundamental market conditions and sentiment-driven fluctuations emanating from outside.

Predicting flash crashes has implications for several stakeholders who are differently affected by extreme events in the market. The major stakeholders who would either gain or lose hugely from flash crashes are investment traders, both retail and institutional. With prices in cryptocurrency moving so fast, traders need to know whether what they are looking at is a temporary dip or a structural decline to make intelligent buy and sell decisions. Government regulators also have an interest in the forecasting and prevention of flash crashes: extreme price disruptions can spread through financial markets overall. Understanding when such events may happen can also, therefore, inform regulators on various policies to lessen the chance of market manipulation and increase investor protection. Main stakeholders will also include virtual cryptocurrency exchanges like Binance and Coinbase, which take the responsibility for maintaining a fair trading environment. Predicting flash crashes allows them to develop safeguard measures like the suspension of trading that can help to prevent cascading liquidation. Algorithmic trading companies and hedge funds also rely on high-frequency trading strategies with automated decision-making. Such firms could detect sudden market anomalies using predictive models and thus dynamically readjust their algorithms and limit exposure to extreme volatility.

Sentiment analysis has also been highlighted as one of the tools helpful in financial predictions. Bollen et al.. (2011) suggested that Twitter sentiment is related to stock market movement, thus providing empirical support for the inclusion of sentiment analysis in financial modeling. In discussions related to cryptocurrencies, Kang et al. (2021) employed SA techniques and found that social media sentiment has a significant effect on short-run price trends. Through the might of VADER and TextBlob for natural language processing, our model will add market sentiment as a feature in the prediction of cryptocurrency price stability by analyzing how public discourse influences it.

While there is useful literature that explains the causes and consequences of flash crashes, our study further integrates anomaly detection, time-series forecasting, and sentiment analysis into one predictive framework. Such a multi-faceted approach ensures that both quantitative financial indicators and qualitative sentiment trends are considered, therefore enhancing the accuracy and robustness of our predictions. This research also helps contribute to more resilient risk management strategies against algorithmic trading, liquidity shortages, and cyber threats in cryptocurrency markets.

**Data & Methods**

**Data**

After deep research for the Bitcoin data, we collected minute level BTC in USD price data using the Bitstamp public API which was free compared to other data resources, covering the period from January 2018 to March 2025\. This high resolution dataset allows us to align sentiment signals from Reddit with actual Bitcoin price movements, helping us explore whether online panic correlates with or even predicts flash crashes. Bitstamp was chosen due to its reliability, reputation, and consistent API access to historical market data, which was a struggle to source compared to other data resources. We explored using other resources to collect our data, but without a paid subscription we could only collect daily price data for the span of the last week. Our analysis required us to have an extensive time span due to the rarity of the flash crash events. In total, the data consisted of over four million rows that represented price per minute for each day from 2018 to the beginning of 2025\. The data includes the following variables for each minute:

* Open: the opening price for the cryptocurrency.

* Close: the closing price for the cryptocurrency.

* High: the high price for the cryptocurrency.

* Low: The low price for the cryptocurrency

* Close: The closing price for the cryptocurrency

* Volume: The total number of trades during the time span

We initially considered using X data (formerly known as twitter) for sentiment analysis, but access was limited due to API restrictions, paywalls, and reduced data availability. As an alternative, we shifted to the Reddit platform, which offered a more open and public source of user sentiment. We were able to filter 40,000 active subreddits, that aligned well with the Bitcoin context and it also includes dedicated communities like r/Bitcoin and r/CryptoCurrency. Our keywords in our search included: Bitcoin, BTC, Flash Crash, Crypto, Cryptocurrency, Bitcoin Crash, Plunge, Collapse, and Crypto Flash Crash. Posts and comments contain detailed, opinionated discussions on market events. Historical Reddit data is freely available through online archives, allowing us to build a large-scale, time-aligned sentiment dataset. The detailed variables in our collected dataset can be seen below: 

* created\_utc: The timestamp in UTC indicating when the post or comment was made by the user.

* subreddit: The name of the subreddit where the content was posted eg: Bitcoin

* title: The title of the post.

* text: The body text of the post or comment.

* score: The number of upvotes the post/comment received.

* Data Volume: The Reddit dataset alone contains over 4 million rows of posts and comments related to Bitcoin, collected from multiple subreddits. This large volume of community-driven content provides rich insights into public sentiment across different time periods and market conditions. 

Note – Our data can be found in the following drive: [https://drive.google.com/drive/folders/1Ni2ihcTp1sF4paDeU4QQTlX8umF5IQ4r?usp=sharing](https://drive.google.com/drive/folders/1Ni2ihcTp1sF4paDeU4QQTlX8umF5IQ4r?usp=sharing)

Next, we will explore some of the descriptive visualizations that we created in our exploratory data analysis. 

The image above is a candlestick chart that provides us with information about Bitcoin prices over a certain period of time. In this image, we are looking at a plot that tells us information about Bitcoin prices around a flash crash on August 8th, 2020\. As we can see this image represents the definition of a flash crash. We see a sudden price drop followed by a swift recovery. The bottom part shows a histogram of the trade volume over this time as well. We can see that volume is pretty constant as the price is dropping, but as soon as the low price occurs, volume starts to increase again. This chart serves as a visual representation to enhance understanding of the issue at hand. 

The following image shows the breakdown of flash crashes over the range of our dataset. As you can see, from 2017 to the beginning of 2018, there were relatively constant crashes. We can see that the downtime between crashes is less than the time between other crashes. Looking at the rest of the plot we see that there were no market events after early 2021\. Looking at late 2018 and mid 2019, we notice that these two time periods experience more flash crashes than the constant time period in 2017 to 2018\. Looking at 2020, we see that the graph spikes with almost 14 flash crash occurrences followed by a few more until early 2021\. This makes sense due to the beginning of the Covid-19 pandemic that led to market instability. 

**Methods**

After pulling the price data using the Bitstamp API, we had to create a target variable for our analysis. We first created a target variable that contained only flash crashes. To identify these points we used the widely acknowledged definition for a flash crash. This constitutes a rapid 3% decline in prices well as a sudden volume spike of around 2.5 times the hourly average. This must be followed by a swift recovery of greater than 60% within ten minutes. For our Reddit data, we utilized VADER to perform the sentiment analysis. VADER is a popular method for sentiment analysis that is commonly used to analyze social media data. Since both the price data and the Reddit data followed the same time breakdown, we were able to combine the datasets relatively easily. We first calculated the sentiment for each observation in the Reddit data, we then grouped these calculations by the timestamp and calculated the average sentiment for each specific time. We were then able to combine our sentiment data to our price data. 

After the data combination was completed we decided to separate our data into training and testing sets using a 70-30 split and performed some exploratory data analysis to gain more understanding about the data. In this analysis, we decided to avoid outlier removal because our analysis mainly focuses on identifying the outliers in the data. We created histograms for each of the numeric variables in the dataset. The variables containing information regarding the price (open, low, high, and close) all followed a relative power distribution. Due to this, we decided to apply a log transformation to these variables. It is worth noting that the volume and sentiment were already centered around zero, so a transformation was not necessary. 

We then explored the class imbalance in our data. There was a significant difference in the number of flash crash occurrences than there were regarding the actual event occurrence. To resolve this issue we applied SMOTE to balance our data. Once our data was preprocessed, we could run our initial model. Based on research and experience, we found that XGBoost was a very popular and effective model for this type of situation. 

After running this model, we did not see good results. We had a very high accuracy of around 99.17%, but we experienced very low values of precision, recall, and F1-score. These results told us that our accuracy was artificially high. This makes sense due to the imbalance of the data. Even though SMOTE had been applied to balance the data, we were still seeing evidence that this was not an effective strategy for this type of analysis. Looking at the breakdown of occurrences in the training and testing data, we were still experiencing a great deal of imbalance in our data. This resulted in our model predicting a high number of non-occurrences and almost no actual flash crashes. 

In order to correct these issues, we added more market events to our data. These events included soft flash crashes and severe crashes. A soft flash crash is defined as a prolonged flash crash. These types of crashes must have at least a 3% decline in five minutes alongside the same volume spike criterion and a recovery within thirty minutes (CoinDesk, 2024). A severe crash is defined by a widely accepted financial benchmark. They are defined as a price decline of at least 20% from the preceding 24 hour high price with less than 50% price recovery over the next seven days (Investopedia, 2024). The addition of these extra events in the data provides our model with more events to predict. This will help to combat the artificially high accuracy we saw in our first model. 

We also decided to attempt to use a different method to balance our data. Instead of using SMOTE, we decided to filter through our data and drop months that have no occurrences of events of interest. In our first attempt, we filtered the data to contain a three to one ratio of 24 hour periods that do not contain a flash crash to 24 hour periods that do contain a flash crash. Using the same sentiment data, we began our next modeling technique. Due to the nature of our data, we decided to avoid using an XGBoost model and instead attempt to train an LSTM Model. Based on our research, we found that these types of models are useful when working with time-series and sequential data (MathWorks). 

Our first model was a standard baseline LSTM, we found that this model performs stably, but struggles with precision. We then created a Bidirectional LSTM. This model did not perform well and overfit the data. Our final model in this sequence was a TCN. This by far was the best performing LSTM variant. It is worth noting that we do not see outstanding performance metrics, but that can be attributed to the rarity of these events. 

At this point, our methodology had been solidified. The next steps of our project aimed at trying to maximize our results. The first step was to filter our data down to a two to one ratio of 24 hour periods that do not contain a flash crash to 24 hour periods that do contain a flash crash. By shrinking our data down even more, we may see more accurate results. We also aimed to explore other sentiment analysis techniques. Instead of using VADER, we decided to try finBERT from hugging face instead. This is a sentiment analysis model that has been trained on exclusively financial text (HuggingFace). We wanted to explore if this method of sentiment analysis improves our results when compared to the VADER method. 

To optimize our BiLSTM model, we tuned key hyperparameters to improve both performance and generalization. We selected a hidden dimension of 256 with 2 LSTM layers and used a dropout rate of 0.3 to reduce overfitting. The learning rate was set to 0.0005, and we trained using a batch size of 128 for more stable gradient updates. To handle class imbalance, we incorporated a pos\_weight adjustment in the loss function. We also implemented a ReduceLROnPlateau scheduler to dynamically lower the learning rate based on validation loss. Finally, we tuned the classification threshold to 0.4 to better balance precision and recall for detecting flash crashes.

We also decided to change the way our sentiment was represented in the data. For both finBERT and VADER. From the original subreddits, we selected the top 10 subreddits that explained the majority of the sentiment in the text. We ran PCA to help with determining this. This worked very well for VADER, but the data for finBERT introduced some missing values due to the Reddit data limits. To solve this issue we rolled up sentiment per minute per subreddit. It is worth noting that we had to reindex the UTC timestamp in the data and calculate a ten minute rolling average to fill in the missing values. After this we split the data using the same split mentioned above and applied the same transformations on the data. We could then use these training and testing sets to train and evaluate our models. 

**Results**

For our analysis, we utilized an LSTM model to make our predictions. After the data was processed, we wanted to focus on comparing the difference between the two sentiment analysis methods that we had used to prepare the data. More specifically we wanted to explore the effect that both VADER and finBERT had when predicting flash crashes in Bitcoin. 

**VADER Results**

| Metric | Precision | Recall | F1-score | Support |
| :---- | :---- | :---- | :---- | :---- |
| Class 0.0 | 0.6379 | 0.6884 | 0.6622 | 22,400 |
| Class 1.0 | 0.7860 | 0.7455 | 0.7652 | 34,392 |
| Accuracy |  |  | **0.7230** | 56,792 |
| Macro Avg | 0.7120 | 0.7169 | 0.7137 | 56,792 |
| Weighted Avg | 0.7276 | 0.7230 | 0.7246 | 56,792 |

**finBERT Results**

| Metric | Precision | Recall | F1-score | Support |
| :---- | :---- | :---- | :---- | :---- |
| Class 0.0 | 0.4565  | 0.9882  | 0.6245  | 22400  |
| Class 1.0 | 0.9682  | 0.2337  | 0.3765 | 34392  |
| Accuracy |  |  | **0.5313**  | 56792 |
| Macro Avg | 0.7123  | 0.6109  | 0.5005 | 56792  |
| Weighted Avg | 0.7664  | 0.5313 | 0.4743 | 56792 |

Despite FinBERT’s domain-specific design, it underperformed with an F1 score of 0.39 and accuracy of 53%. This was due to a mismatch between FinBERT’s formal financial language training and Reddit’s informal, slang-filled content, leading to noisy sentiment signals and overfitting. In contrast, VADER—though lexicon-based—was better suited for short, social media text. When paired with BiLSTM, it achieved a stronger F1 score of 0.69 and accuracy of 67.5% eventually leading to around 73%. Its simplicity and alignment with Reddit’s tone allowed better generalization, proving that context-appropriate tools, even simpler ones, can outperform complex models in real-world scenarios.

While working with both models, we found that FinBERT, despite its domain-specific financial embeddings, struggled to align well with actual outcomes—our evaluation showed an RMSE of **0.7344** and MSE of **0.5393**, pointing to larger prediction errors. On the other hand, VADER, though much simpler, consistently produced more stable results. While we didn’t explicitly compute RMSE/MSE for VADER, the improvement in F1-score and accuracy made it clear that it generalized better and handled Reddit’s informal sentiment more effectively. It was a clear reminder that sometimes, simplicity paired with context-fit beats sophistication.

The VADER-based model demonstrated a stronger balance between detecting crashes and avoiding false positives, correctly identifying **26,894 flash crashes** (TP) while maintaining a much lower false negative rate (7,498) compared to FinBERT’s 26,356. This matrix clearly shows VADER’s superior recall and generalization in capturing actual flash crash events from social media sentiment.

Why Vader might be better than Finbert ?

| Aspect | Aspect | FinBERT |
| ----- | ----- | ----- |
| **Recall (Crash Class)** | High recall (0.61) – captured most crash events | Very low recall (0.24) – missed majority of crashes |
| **F1 Score** | Strong F1 (0.72) with balanced precision and recall | Weak F1 (0.53) due to class imbalance and also after finbert the sentiment scoring |
| **Threshold Robustness** | Performed well even with minimal tuning | Required exhaustive threshold sweep for acceptable output |
| **Strengths** | ✦ High precision ✦ Rich embeddings | ✦ Less GPU intensive ✦ Poor sentiment for social media comments related to bitcoin  |

**Discussion**

Based on our results, we see that our accuracy is significantly higher than other metrics such as precision and recall. This is due to class imbalance. Even though we balanced our data with a two to one ratio of 24 hour periods that do not contain a flash crash to 24 hour periods that do contain a flash crash, there was still a significant imbalance of zeros to ones in the data. This has resulted in a large number of true negative predictions in our output. The abundance of true negatives resulted in an artificially high accuracy. We are interested in the precision and recall values. Recall tells us the proportion of all flash crashes that were correctly predicted as being flash crashes and the precision tells us proportion of flash crashes that were truly flash crashes. Looking at our results, we were surprised with how high our values of precision and recall were. Initially, we really struggled with the size and imbalance of our data, but by decreasing the number ratio of non-events to events, we were able to produce results that were acceptable. With the values of precision and recall that we achieved for our model, investors as well as financial firms have a relatively reliable way to determine if a flash crash has or will occur. With this model and knowledge, these stakeholders can invest more safely and appropriately by avoiding situations where they will lose assets in a trade. 

**Limitations**

When we first began our project, we assumed that our data, especially the price statistics, would be easily accessible. After our initial search, we discovered that many of the sources that offered this data required a paid monthly or yearly subscription to access the data for an extended period of time and the time increments that we needed. We eventually had success utilizing the Bitstamp API to access the data we needed, but this provided us with limited descriptive statistics. We would have liked to have more such as Ethereum, Tether, and XRP, but these variables were not realistically accessible for the purpose of our project. It is also worth noting that the addition of more symbols would have greatly increased the file size of our data. We had already been having some issues with uploading our data to both our github repository and our virtual machine. Increasing the size of the data would only increase the difficulties we faced. 

We also faced an issue of class imbalance. Flash crashes are not a relatively constant occurrence in cryptocurrency. They do happen, but it is not a common occurrence. Due to this, we had many more instances where a flash crash did not occur compared to the positive occurrences. To resolve this issue, we had to remove some of the cases where a flash crash did not occur as documented in the methodology section. This was not optimal as we would have much rather preferred to use the full dataset, but this was not plausible. Our model on the entire dataset rarely predicted positive flash crash occurrences, which resulted in performance metrics that did not provide us with much information regarding our research question. 

**Future Work**

This analysis only focuses on Bitcoin cryptocurrency. There are many other cryptocurrencies investors buy and sell. While Bitcoin is the most popular cryptocurrency for investors, we would live to explore flash crashes and market movement in other popular cryptocurrencies such as Ethereum, Tether, and XRP. We believe it would be really interesting to apply our analysis to these other currencies to see if they follow the same patterns. In other words, we would like to explore if these other cryptocurrencies experience flash crashes when Bitcoin does and if they follow the same trends. We also believe it would be interesting to explore some more independent variables in our analysis. Our model heavily relies on sentiment analysis and we would like to explore if adding additional market indicators to our model would improve our results. A lot of these features are not publicly available, so in order to accomplish this, we would have to reach out to an industry leader such as Binance to acquire these additional statistics. 

We would also like to filter down the time increment between observations to see if we can get better results. We would like to explore how far in advance our model can predict a flash crash. We initially planned on doing this in our analysis, but it proved to be too much to complete before the project due date.

**References**

* Akyildirim, E., Sensoy, A., & Söylemezgil, S. (2020). Flash Crashes in Cryptocurrency Markets and the 2019 Kraken Bitcoin Flash Crash. *Journal of Financial Markets*.  
* Bollen, J., Mao, H., & Zeng, X. (2011). Twitter mood predicts the stock market. *Journal of Computational Science, 2(1), 1-8*.  
* Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. *Journal of Econometrics, 31(3), 307-327*.  
* Feuerriegel, S., & Gordon, J. (2018). Deep learning for detecting financial statement fraud. *Decision Support Systems, 116, 38-50*.  
* Fischer, T., & Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions. *European Journal of Operational Research, 270(2), 654-669*.  
* Gai, J., Lin, J., & Ricketts, P. (2022). Machine Learning for Flash Crash Prediction: A Data-Driven Approach.*Finance Research Letters, 47, 102654*.  
* Johnson, N. F., Zhao, G., Hunsader, E., Meng, J., Ravindar, A., Carran, S., & Tivnan, B. (2013). Abrupt rise of new machine ecology beyond human response time. *Scientific Reports, 3(1), 2627*.  
* Kang, W., et al. (2021). Sentiment Analysis for Cryptocurrency Markets Using VADER. *Financial Data Science Journal*.  
* Kenton, W. (n.d.). Flash crash: Definition, causes, history. *Investopedia*. [https://www.investopedia.com/terms/f/flash-crash.asp](https://www.investopedia.com/terms/f/flash-crash.asp).  
* Oosthoek, K. (2021). Flash Crash for Cash: Cyber Threats in Decentralized Finance. *arXiv preprint arXiv:2106.10740*.  
* Wehrli, A., & Sornette, D. (2022). Classification of Flash Crashes Using the Hawkes (p, q) Framework. *Taylor & Francis*.  
* Xing, F., Cambria, E., & Welsch, R. (2021). Natural Language Processing for Financial Text Analysis. *IEEE Computational Intelligence Magazine, 16(2), 48-58*.  
*  CoinDesk. (2020). “Bitcoin Plunges Nearly 39% on Black Thursday Crash.” Retrieved from https://www.coindesk.com/markets/2020/03/12/bitcoin-plunges-nearly-39-on-black-thursday-crash/.  
* Investopedia. (2024). “Bear Market.” Retrieved from https://www.investopedia.com/terms/b/bearmarket.asp.  
* Investopedia. (2024). “Flash Crash Definition.” Retrieved from https://www.investopedia.com/terms/f/flash-crash.asp.  
* CoinDesk. (2024). “Understanding Flash Crashes in Cryptocurrency Markets.” Retrieved from https://www.coindesk.com  
* GeeksforGeeks. (2024, December 11). *Sentiment analysis using VADER – Using Python*. GeeksforGeeks. [https://www.geeksforgeeks.org/python-sentiment-analysis-using-vader/](https://www.geeksforgeeks.org/python-sentiment-analysis-using-vader/)  
* MathWorks. (n.d.). *What is long short-term memory (LSTM)?* MATLAB & Simulink. [https://www.mathworks.com/discovery/lstm.html](https://www.mathworks.com/discovery/lstm.html)  
* ProsusAI. (n.d.). *FinBERT: Financial sentiment analysis with pre-trained language models*. Hugging Face. [https://huggingface.co/ProsusAI/finbert](https://huggingface.co/ProsusAI/finbert)

