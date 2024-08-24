---
layout: post
author: Name
title: "Applied Data Science Project Documentation"
categories: ITD214
---
## Project Background
The dataset, obtained from Kaggle, related to conversion record of inquirers for enrolling course in X Education, an institution offering a diverse range of courses including financial management, e-commerce, business administration, healthcare, IT project managements, service excellence and other courses.
![image](https://github.com/user-attachments/assets/93f3e825-f877-48cb-a7af-f1eae5d7eac8)

Our team got 4 members and we have our own objective analyzing the same dataset above. The respective objectives are as follows:
1. To analyze the behaviors of candidates landing on various online platforms, an institution that markets courses on multiple websites examines the factors influencing potential candidate conversion success.
2. To examine the factors that lead to more time being spent on the website, and as such they can spend more resources to target this group of people that is likely to lead to successful conversion.
3. To identify strengths and weaknesses for improving conversion rates through comment analysis.
4. Studying external factors affecting the successful selection, ‘Converted’, we may establish the following four hypotheses and uncover the hidden potential from the researches:
4.1 Hypothesis A: There is a regression relationship between Converted rate and the various Marketing or Media dimensions.
4.1.1 Model 1.1 is on the Boolean columns, and
4.1.2 Model 1.2 is on the numerical columns
4.2 Hypothesis B: The Converted rate can be predicted from clients’ feedback and their training and professional background. 
4.3 Hypothesis C: There are opportunities to extend the markets to other major cities and overseas
4.4 Hypotheses D: The reviews can help predicting from the review-related surveys and reviews in free text may predict the subject’s decisions
4.4.1 Model 4.1 is on the text analytics to uncover the candidates’ feeling or hidden expectation from their reviews.
4.4.2 Model 4.2 tried to uncover the successful conversion Vs their reply on the Asymmetrique Activity and Profile.

I take charge of the Objective 4.

In analyzing the successful factors for any online platforms to recruit in the courses, the institute collected 9,240 samples consisting of 37 data columns or features.  Other than the 2 identification columns and the column whether the candidate convert or not, those other columns could be grouped into a few categories, namely Marketing (18 columns), Demographic (2 columns), About the Client (6 columns), Reviews (6 columns) and 3 numerical columns for counting purposes.  We have to design structural approaches on those features in order to dig out the hidden information to assist the development plans for the institution.


## Work Accomplished
Hypothesis A: 
From the headmap below, ‘Magazine’, ‘Receive More Updates About Our Courses’, ‘Update me on Supply Chain Content’, ‘Get updates on DM Content’ and ‘I agree to pay the amount through cheque’ are highly corelated and thus not helpful for this analysis. Columns ‘TotalVisits’, ‘Total Time Spent on Website’ and ‘Page Views Per Visit’ are stood out by themselves and could be studied as a grouping themselves.
![image](https://github.com/user-attachments/assets/0c0904e5-659f-4bb8-b3e1-446ea5279138)
Two Logistic Regression Models, Model 1 and Model 2 are to be built with variables set up as illustrated by the following table.
![image](https://github.com/user-attachments/assets/8a8cb38e-fe99-4bca-8aa4-3939171f792f)
For both Models: to employ the ‘k’ Logistic Regress Model using a One-vs-Rest (OvR) approach initially.  In case the recall for Class 1 is poor, adjusting parameters with GridSearchCV approach and/or further Hyperparameter tuning via experiments with different thresholds will be implemented. To avoid overfitting, fine-tuning a little bit on scoring=’f1’ for the adjusted threshold Confusion Matrix might be required.

Hypothesis B: 
Data exploring revealed that 78% of the missing inputs in column ‘How did you hear about X Education’ and thus it is better to drop it from further analysis.  
![image](https://github.com/user-attachments/assets/0d5ec9c9-75c7-405d-852c-7c5569b0f37e)

The rest of the columns are good for converting to dummies columns and apply the ‘k’ Logistic Regression Model with the One-vs-Rest (OvR) approach.  Structure of the Model is as follows:
![image](https://github.com/user-attachments/assets/4d0742fc-7425-48c3-b285-9c9a43945cb6)
After the Logistic Regression with scikit-learn model, we may further use the Random Forest Model, Random Forest Model and Gadient Boosting Model to compare the performance.  From the regression formula generated, we may expect to check the important features from all the said models.

Hypothesis C: 
Based on the data exploration per below, it seems that there is not much to be able to drill in and thus no analysis will be conducted. Further analysis is definitely required to make use of other proper columns but my team-mate is working on this topic and thus I would leave it to him for deeper study.  However, for further developments, the result demand some growth strategies to be developed: 
Retention: keep improving the local market and that in Mumbai as a base.
Penetration: Extending to cities other than Mumbai and countries other than India.
Conversion: Extending to other industries.
![image](https://github.com/user-attachments/assets/554bf93f-2dcd-408d-82ef-e9a2b4b56d2d)

Hypothesis D: 
Model 1 is on the text analytic on two free text columns, ‘What matters most to you in choosing a course’ and ‘Tags’.  To prepare for this analysis, the two columns are to be concatenated into one column.  I then used Scikit-learn’s CountVectorizer to convert a collection of text documents to a vector of term/token counts.  The Model to be applied is the Vector Space Model using the most popular statistical measure TF-IDF to evaluate the relevance of words to the document frequency of the word across a set of documents.  The Singular-Value Decomposition method (SVD) is then used to simplify the calculations by reducing a matrix to its constituent parts.  2 components were chosen and then the scatter plots of the two principal components and points against LSA principal components were obtained to check the semantical similarity. My matrix is then projected/ reduced from 52 to 2 dimensions. The pre-trained model used the common ‘GoogleNews-vectors-negative300.bin’ and then train my own Word2Vec.  As the GoogleNews … bin is too big to load in my computer, I only set the limit to 500,000. Afterward, my predictions can be conducted accordingly.
Model 2 is to use the Logistics Regression Model with scikit-learn to check whether the conversions rate got anything related to the structural data columns, 'Asymmetrique Activity Score', 'Asymmetrique Profile Score', 'Asymmetrique Activity Index', 'Asymmetrique Profile Index'.  A conversion from the input of Low/Medium/High to numbers is required.  I chose to convert to -1/0/1 instead of the usual 1/2/3 so that some of the effect might be offset each other.
![image](https://github.com/user-attachments/assets/52e5d0a2-5e35-4cbf-aaff-d96225b04265)

### Data Preparation
Hypothesis A:

For Model 1: 
1.	Converting columns with inputs Yes/No to 1/0 for the regression analysis.
2.	Checking rows with null entries and remove, if any. 
3.	Split the datasets into 70%/30% for training and testing respectively.
4.	Verifying the confusion matrix to for the acceptable Recall % for Class 1.
5.	If not acceptable, adjusting the parameters in order to obtain the best option.
6.	Try other models like Support Vector Machine (SVM) or Ensemble method for comparing the effectiveness of the predictions.
7.	Compute the regression formula using the model with the optimal parameters.

For Model 2:
1.	Checking the null entries: only ‘TotalVisits’ column got 137 rows with missing entries
2.	Fill such rows with the means value
3.	Split the datasets into 70%/30% for training and testing respectively. 
4.	Verifying the confusion matrix to for the acceptable Recall % for Class 1.
5.	If not acceptable, adjusting the parameters in order to obtain the best option.
6.	Try other models like Support Vector Machine (SVM) or Ensemble method for comparing the effectiveness of the predictions.
7.	Compute the regression formula using the model with the optimal parameters.

Hypothesis B:
1.	Logistic regression model would still be suitable between the ‘Converted’ column and the rest. I need to convert the categorical features to dummies columns and thus the columns with 0/1 values could be converted before applying for the use in the model. The suitable mode is still the Regression Model. As a start, I used the ‘k’ Logistic Regression Model with the One-vs-Rest (OvR) approach. 
2.	The categorical features I chose was the ‘Specialization’ column.  Hence, I separated out the ‘Specialization’ column first and first convert the rest of the subject columns into dummies columns and then add back the ‘Specialization’ column into the DataFrame.
3.	As the input for ‘Converted’ is only 0 or 1, the simple Binary Classification model with scikit-Learn was used which is also useful and easier to fit for the regression model.
4.	Data in the ‘Tags’ column will be studied under Hypothesis 4 via Text analysis whereas specialization will be used with the current occupation as clients’ background to research.

Hypothesis C:

Only data exploring stage has been done and it is found that we could not do much for the further analysis unless combining with other features/ columns.  As my team member will be working on this demographic area.  I would stop at the data exploring level.
![image](https://github.com/user-attachments/assets/c08aef63-adeb-4e65-9c97-dd714ed53b50)

Hypothesis D:

Model 1:
I did usual data cleaning by removing stop words, numbers and punctuations, and domain specific common words using NLTK.  From the above data exploring, after the usual stop words removal, the word cloud still showed words and punctuations like ‘nan’, ‘(‘ and ‘)’ appeared in the top 20 characters list.  As such, I created a common word to add in the stopword dictionary and remove the punctuations and the word ‘nan’ additionally.  The stemming was by PorterStemmer.  My concatenated column was concatenated into a single line of text and thus it may be straightly applied.
The top-most and least frequently occurring words are as follows and the keywords looks good and relevant:
![image](https://github.com/user-attachments/assets/ea84e3fb-7734-4805-a9b8-cb65b3935f87)
From the frequency plot of the words, the unexpected words were removed and probably at most 15 words is enough to dominate the major proportions of my model.  The next word cloud might confirm the same. It shows that the keywords, better, career, prospect, reading, email, ringing, interested, courses, etc. which are familiar if we look at the important features from Hypothesis B.  The results are quite consistent.
![image](https://github.com/user-attachments/assets/6bac032d-e1b0-4fdf-b7ad-ee26286c300c)
![image](https://github.com/user-attachments/assets/76c648c0-91d2-4d05-8ab6-208bb60580d4)

Model 2:
1.	Converting columns with inputs Low/ Medium/ High to -1/0/1 for the regression analysis.
2.	Checking the null entries: there were 4,218 rows with missing entries or 45.6% of the total counts
3.	Anyway, I removed rows with null inputs and conduct the regression analysis.
4.	Split the datasets into 70%/30% for training and testing respectively.

### Modelling
Hypothesis A:

Model 1:

Poor recall % in the result using the Logistic Regression Model and thus adjusted parameters forcing the better recall was applied.  The result from the Adjusted Threhold Confusion Matrix is shown below:
![image](https://github.com/user-attachments/assets/36bbd73c-bdd8-4e2a-b3cb-5053d64c4167)

Model 2:

Poor recall % in the result using the Logistic Regression Model and thus adjusted parameters forcing the better recall was applied.  The result from the Adjusted Threhold Confusion Matrix is shown below:
![image](https://github.com/user-attachments/assets/78b19f84-a11f-4d26-9b54-b9a27681bd14)

Hypothesis B:

The Confusion Matrix is good for this model. I used more models and compare the results as follows:
![image](https://github.com/user-attachments/assets/7ec550c6-a180-4e5a-9520-303f7c488876)

Hypothesis C:
Nil

Hypothesis D:

Model 1:

The groupings are considered to be closer together with only a few outliners in the plot.  When I tried to change the components to 3, the scatter plots become messy.
![image](https://github.com/user-attachments/assets/6e913716-4a0f-4c39-a2c5-a70e8acd1a55)

Model 2:

Poor recall % in the result using the Logistic Regression Model even with experiments by adjusting the parameters using GridSearch to find the best option, I then tried to include additional hyperparameters like penalty, fit_intercept, and class_weight to explore more combinations hoping to improve the model performance.  The result turned out to be overfitted.  Even though I adjusted the parameters to avoid the recall = 1, there is no improvement.  As such, I decided to terminate this part of the analysis.  In fact, my teammate who did another approach of analytics, also chose to drop these four columns too.
![image](https://github.com/user-attachments/assets/e6009d94-e79c-4437-9274-324b05377e0b)

### Evaluation
Hypothesis A:
Model 1:
The adjusted Threshold Confusion Matrix got 0.71 for the Recall on Class 1 suggusts that the model for the prediction is quite good.  However, accuracy of 0.47 is low.  The data collection on the Yes/No survey usually got more errors including the typo or respondents easily to anyhow input or influence by peers, etc.  The poor results indicates that further developments are required.  Anyway, for completeness, the regression formula below may still form some references.  For example, the weightings might show some importance of the features.  The positive or negative weightings are the other indicators.
![image](https://github.com/user-attachments/assets/9a6489d1-82a7-49d0-a74f-e9f01e5b62e8)
ROC curve is plotted for the True Positive Rate (TPR) against the False Positive Rate (FPR) at various threshold settings to provide a way to visualize the trade-off between the sensitivity (or recall) of the model and the fall-out (1 - specificity) as you adjust the decision threshold. The ROC curve is only slightly above 0.5, which is the performance of a random classifier.  I compared the Logistic Regression, Random Forest and Gradient Boosting but only the Gradient Boosting shows the plot.
![image](https://github.com/user-attachments/assets/2fee8ac4-f4ca-47b4-822e-df34390f15ca)
Model 2:
As this result shows that the model is more reliable for using for predictions.  We may then compute the regression formula as follow and may draw some indications from the weighting that the total visits is two times more important than the time spent on websites and page views per visit would be negatively impact to the converted rate (Re: Headmap below).
![image](https://github.com/user-attachments/assets/5a388edf-dd17-47f2-8de1-675021c4120b)
Again, ROC curve is plotted. The model whose ROC curve is closer to the top left corner has a better true positive rate for a lower false positive rate, indicating a better classifier.  I compared the Logistic Regression, Random Forest and Gradient Boosting and the Gradient Boosting turns out to be the best amongst these three models.
![image](https://github.com/user-attachments/assets/6044c8ca-b191-4a8b-8084-3bd5bf3b7499)

Hypothesis B:

The regression formula is computed below also.  Now, we can see lots of features reflecting from the formula and I further processed the feature importance charts for the top 10 features using Random Forest Model and Gradient Boosting Model as well.
![image](https://github.com/user-attachments/assets/96544f3a-149c-42e0-bc1a-85ddd64a588b)
![image](https://github.com/user-attachments/assets/8ab3a03d-4a54-4f76-bc2a-2c57ee2b1aa5)
![image](https://github.com/user-attachments/assets/80c5533d-4b05-490e-ac81-c678f2f6f583)
![image](https://github.com/user-attachments/assets/f6d57f88-fde1-43bc-b394-36b9ffa9de03)
Noticed that the weighting provides the positive and negative impacts to the conversion rates.  We may plot the top 10 positive weightings and the top 10 negative weightings below.  From the marketing perspectives, we have the top 10 factors to promote and also the top 10 factors to avoid but the Random Forest and the Gradient Boosting seems to mix them together.
![image](https://github.com/user-attachments/assets/42b6183d-7f1a-4b31-947c-a3d5f53db010)
![image](https://github.com/user-attachments/assets/0a3bc61c-ed26-4271-bada-4bb18a717e7c)
Alternative observations for the positive and negative correlation is from the heatmap. Based on the above findings, I highlighted them on the headmap and found that certain consistencies could be seen.
![image](https://github.com/user-attachments/assets/94ce3bab-b045-446b-9dd9-75e7a7fb7527)

Hypothesis C:
Nil

Hypothesis D:
Model 1: Text Analytic
I used two training datasets in the model, the GoogleNews-vectors-negative300.bin and from my dataset.  Following predictions using a same word from the two models, the predictions showed that my own model can predict words more up to my expectations.  For testing purpose, I also used another word from my model which also shows the expected words similar to my dataset.
Predictions from GoogleNews-vectors-negative300.bin:
![image](https://github.com/user-attachments/assets/f52b8be7-b26f-45ed-8f40-2b5637bbca35)
Predictions from my trained model:
![image](https://github.com/user-attachments/assets/174967c1-731a-4b27-9888-eacb0afdc2ab)
![image](https://github.com/user-attachments/assets/a47249e0-f231-4473-bcfb-cd20f8a4fe9f)

Model 2: Regression Analysis 
Nil

## Recommendation and Analysis
Identifying the Marketing issues

The model on the Visits indicators is good meaning that the converted rate is mainly proportional to the Total Visits and then the Total Time Spent on Website but negatively related to the Page Views Per Visit.
The converted rate Vs the media used did not get a good model but the prediction on the successes is still good at 71% and we had the ‘Through Recommendations’ and ‘Newspaper Articles’ being the higher weighting of the regression formula whereas ‘X Education Forums’ and ‘Digital Advertisement’ provided the otherwise.
From the client columns, the model built got good fit and resulted in features importance outcome.  The important feature suggests ‘will revert after reading email’, ‘Better Career Prospects’, ‘Finance Management’ and ‘Online Search’, another keywords, ‘Already a student’ and ‘Closed by Horizon’ as the negative side.  Those areas suggest the positive and negative focuses that must be addressed.
The Demographic research concluded that the market is only in India and mainly Mumbai.  In order to extend its business beyond Mumbai and India, there are lots of potentials.  They may start to think of the common growth strategic on the Retention, Penetration beyond Mumbai to other India cities and the Conversion plan to extending to other Countries and beyond their current industries.
Text Analytic is a powerful tool to understand the emotional aspects of the candidates.  The model confirmed some of the observations above.  The dataset now is now big enough to build up excellent model.  With the transfer learning models, that training should not be an issue as time goes by.

Recommendations

The missing data issue is serious and thus the institution should improve the quality of the data collection, also need to have clarity on the terminologies.  Redesign of the survey might be also essential.
To data collection should extend to the Sentiment Analysis and more reviews in the form of free text type should be encouraged to enrich the text library so that the model training can be more reliable.

## AI Ethics
This report based on a few predictive analytics models which might affect the decisions of the management, which in turn, the change in policy or any marketing or growth strategies, in terms of the privacy, fairness, accuracy, accountability and transparency, could be important for maintaining integrity and trust.  By publishing the report, I only aim at creating some platform to discuss the result academically without any intension to diverse or violate the ethical standards that I ought to maintain and respect.

## Source Codes and Datasets
(https://github.com/RatseyChing/itd214_proj)
Sources of the datasets is: https://www.Kaggle.com/code/ggsri123/lead-score
