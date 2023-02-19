# Sampling
Sampling refers to the process of selecting a subset of data from a larger dataset to use in the training or evaluation of a machine learning model. The goal of sampling is to capture the essence of the larger dataset while reducing the computational cost and improving the generalization performance of the model.

## Methodolgy
Using a dataset for the detection of credit card fraud, we are using five different sampling approaches in this and we will further examine their accuracy and go over the outcomes.
### 1. Imbalanced dataset to balanced dataset conversion: 
The class "1" has less samples in the provided dataset. By oversampling instances of class "1" and setting them equal to instances of class "0," we were able to resolve this problem.
   
### 2. Generating samples using five different sampling techniques: 
   1. Simple Random Sampling : Simple random sampling is a statistical technique where every member of the population has an equal chance of being selected as part of the sample. This method involves selecting a sample at random from the entire population without any specific criteria or considerations.

   2. Systematic Sampling : Systematic sampling is a technique where members of the population are selected at regular intervals. For example, a researcher might choose to sample every 10th person from a list of names. This method is relatively easy to use and can be efficient, but it may introduce bias if there is a pattern in the list.
      
   3. Stratified Sampling : Stratified sampling is a technique where the population is divided into groups or strata based on specific characteristics, and a random sample is selected from each stratum in proportion to its size. This method is often used to ensure that the sample is representative of the population with respect to certain characteristics.
      
   4. Cluster Sampling : Cluster sampling is a technique used when it is difficult or impossible to obtain a complete list of members of the population. In this method, the population is divided into clusters or groups, and a random sample of clusters is selected. Then, all members of the selected clusters are included in the sample.
 
   5. Convenience Sampling : Convenience sampling is a non-probability sampling technique where participants are chosen based on their availability or willingness to participate. This method is often used in situations where it is difficult or impractical to obtain a representative sample, such as when conducting research in a public space. However, convenience samples are generally not representative of the population, and the results may not be generalizable to other contexts.
### 3. Machine Learning models applied on above five sampling techniques: 

   - K-Nearest Neighbour
   - Support Vector Machine(SVM)
   - Random Forest Classifier
   - Logisitic Regression
   - Decision Tree Classifier

### 4. Result 
|        Models &darr;/Sampling&rarr;       | Simple Random | Systematic | Cluster | Stratified | Convenience |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Logistic Regression  | 93.814433 | 80.0 | 91.666667 | 90.526316 | 98.0 |
| SVM  | 67.010309 | 60.0 | 67.12963 | 74.736842 | 99.0 |
| KNN  | 88.659794 | 40.0 | 95.833333 | 94.736842 | 99.0 |
| Decision Tree | 92.783505 | 50.0 | 93.981481 | 94.736842 | 99.0 |
| Random Forest | 100.0 | 90.0 | 100 | 100.0 | 99.0
   
   
## Conclusion
 Random Forest Classifier is giving the best result in each case..!