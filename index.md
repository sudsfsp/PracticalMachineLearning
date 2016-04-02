---
title: "Practical_ML"
author: "Sudhakar"
date: "April 1, 2016"
output: html_document
---
**Title : Practical Machine Learning-Project**
=============================

**Summary:**
===============
The practical ML project focused on creating accurate and complex model for predicting output variable. We are going to see how ML algorithm solves real life situation and problem with the help of complex model.  The project uses data from the Weight Lifting Exercises (WLE) Dataset. The approach we propose for the Weight Lifting Exercises dataset is to investigate "how (well)" an activity was performed by the wearer.  In this process, a group of enthusiasts who take measurements about themselves regularly to improve their health.  In this project, The exercises were performed by six male participants aged between 20-28 years, and they were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions, identified as classes A, B, C, D and E. Class A corresponds to a correct execution of the exercise, and the remaining four classes identify common mistakes in this weight lifting exercise. (See the section on the [Weight Lifting Exercise Dataset](http://groupware.les.inf.puc-rio.br/har)  The goal of project is to predict the manner in which they did the exercise using the accurate Machine Learning algorithm.  The following analysis uses a Decision Trees (rpart) and random forest prediction algorithm for predicting manner of exercise performed by user.  The results of the analysis confirm that the random forest algorithm model achieves high prediction accuracy.


**Data Source:**
==================

The training data for this project are available here:[pml-training.csv](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)    
The test data are available here:[pml-testing.csv](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)    

**Downloading the datasets:**
===============================

The following code download the training data and testing data automatically.The codes generate dataML folder where you can see all data in your working directory(check getwd()).   


```r
trainUrl ="https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testUrl = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

train_data = "./dataML/pml-training.csv"
test_data = "./dataML/pml-testing.csv"

if (!file.exists("./dataML")) {
  dir.create("./dataML")
}
# if files does not exist, download the files
if (!file.exists(train_data)) {
  download.file(trainUrl, destfile=train_data)
}
if (!file.exists(test_data)) {
  download.file(testUrl, destfile=test_data)
}
```

**Package Requirement:** 
=========================

The following code work as install automatically required package for run below code smoothly.  

```r
list.of.packages <- c("dplyr", "caret", "rpart", "randomForest","VIM","rattle")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)
```

**Loading libraries:**
========================


```r
suppressMessages(library(caret))
suppressMessages(library(rpart))
suppressMessages(library(randomForest)) 
suppressMessages(library(dplyr))
suppressMessages(library(rattle))
suppressMessages(library(VIM))
```

**Data Exploration:**  
=======================

There are many missing values. These missing values come in two versions: the usual NA value, but also as values of the form "#DIV/0!". 

```r
pml_train <- read.csv("./dataML/pml-training.csv", header=T, comment.char = "",
                      na.strings=c("","#DIV/0!","NA", " "))
pml_test <- read.csv("./dataML/pml-testing.csv", header = T,comment.char = "",
                     na.strings=c("","#DIV/0!","NA"," "))

dim(pml_train)
```

```
## [1] 19622   160
```

```r
dim(pml_test)
```

```
## [1]  20 160
```

**Processing and Cleaning the Data set:**
=====================================

The following del_Na function delete columns in the data set where the sum of  missing data (NA value) in coloumn is greater than 95%(because it is not easy to imputation) as well as dropped first to seven variable from the data set because these variable is not useful in the sense of following analysis.   


```r
del_NA <- function(x){
        aggr_plot <- aggr(x,plot = F, numbers=TRUE,sortVars=TRUE,gap=3)
        check_miss <- data.frame (aggr_plot$missings)
        miss_pct <- mutate(check_miss, Count=(Count/nrow(x))*100)
        miss_data <- subset(miss_pct, Count > 95.00)
        miss_names <- miss_data$Variable
        training_df <- x[, !(names(x) %in% miss_names)]
        ## First to seven variable is not useful in data set. 
        training_df <- training_df[,-c(1:7)]
        return(training_df)
}
new_data_train <- del_NA(pml_train)
new_data_test <- del_NA(pml_test)
dim(new_data_train)
```

```
## [1] 19622    53
```

```r
dim(new_data_test)
```

```
## [1] 20 53
```

```r
table(new_data_train$classe)
```

```
## 
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```

**Non zero Variance:**
===========================

We also try to remove variables with nearly zero variance but all variables show their variance greater than zero. So we keep all variables.

```r
nsv <- nearZeroVar(new_data_train, saveMetrics = T )
head(nsv,10)
```

```
##                  freqRatio percentUnique zeroVar   nzv
## roll_belt         1.101904     6.7781062   FALSE FALSE
## pitch_belt        1.036082     9.3772296   FALSE FALSE
## yaw_belt          1.058480     9.9734991   FALSE FALSE
## total_accel_belt  1.063160     0.1477933   FALSE FALSE
## gyros_belt_x      1.058651     0.7134849   FALSE FALSE
## gyros_belt_y      1.144000     0.3516461   FALSE FALSE
## gyros_belt_z      1.066214     0.8612782   FALSE FALSE
## accel_belt_x      1.055412     0.8357966   FALSE FALSE
## accel_belt_y      1.113725     0.7287738   FALSE FALSE
## accel_belt_z      1.078767     1.5237998   FALSE FALSE
```

**Model Building:**
======================

Partitioning the Training data set (Validation-set Approach):
===============================================================

Following the usual practice in Machine Learning, we will split our data into a training data set (75% of the total cases) and a testing data set (with the remaining cases). This will allow us to estimate the out of sample error of our predictor. We will use the caret package for this purpose, and we begin by setting the seed to ensure reproducibility.


```r
set.seed(123)
inTrain <- createDataPartition(y=new_data_train$classe,p=0.75, list=F)
training <- new_data_train[inTrain,]
testing <- new_data_train[-inTrain,]
```

Machine Learning Algorithm - Decision Tree classification :
===========================================================

We are going to use our first model as Decision Tree in ML (Machine Learning). Decision Trees (DTs) are a non-parametric supervised learning method used for classification. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.


```r
## Using caret package for create decision model. 
set.seed(123)
modfit <- train(classe~., method='rpart', data=training)
## plot the decision tree
fancyRpartPlot(modfit$finalModel)
```

![alt tag](https://raw.githubusercontent.com/sudsfsp/PracticalMachineLearning/b4cc499d11328bea867e083be82316f556448401/PNG/plot1.png) 

```r
## predict decision tree model on test data
pred <- predict(modfit, newdata=testing)
confusionMatrix(pred, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1262  378  418  356  144
##          B   20  312   26  141  101
##          C  107  259  411  307  248
##          D    0    0    0    0    0
##          E    6    0    0    0  408
## 
## Overall Statistics
##                                           
##                Accuracy : 0.488           
##                  95% CI : (0.4739, 0.5021)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3307          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9047  0.32877  0.48070   0.0000  0.45283
## Specificity            0.6307  0.92718  0.77254   1.0000  0.99850
## Pos Pred Value         0.4934  0.52000  0.30856      NaN  0.98551
## Neg Pred Value         0.9433  0.85200  0.87570   0.8361  0.89020
## Prevalence             0.2845  0.19352  0.17435   0.1639  0.18373
## Detection Rate         0.2573  0.06362  0.08381   0.0000  0.08320
## Detection Prevalence   0.5216  0.12235  0.27162   0.0000  0.08442
## Balanced Accuracy      0.7677  0.62797  0.62662   0.5000  0.72567
```

**Note:** The Output result show that decision tree capture only 49% accuracy in the classification model. Let's try another method in dicision tree model. We will use "Class" (classification tree model) method in base rpart model.   


```r
## using base rpart model with "class"" method.
set.seed(123)
tree_model <- rpart(classe~., data=testing, method='class')
pred <- predict(tree_model, newdata=testing, type='class')
confusionMatrix(pred, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1211  153   10   58   17
##          B   69  522   54   36   55
##          C   51  153  729  207   75
##          D   48   70   27  430   66
##          E   16   51   35   73  688
## 
## Overall Statistics
##                                           
##                Accuracy : 0.73            
##                  95% CI : (0.7174, 0.7424)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6582          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8681   0.5501   0.8526  0.53483   0.7636
## Specificity            0.9322   0.9459   0.8800  0.94854   0.9563
## Pos Pred Value         0.8357   0.7092   0.6000  0.67083   0.7972
## Neg Pred Value         0.9467   0.8976   0.9658  0.91227   0.9473
## Prevalence             0.2845   0.1935   0.1743  0.16395   0.1837
## Detection Rate         0.2469   0.1064   0.1487  0.08768   0.1403
## Detection Prevalence   0.2955   0.1501   0.2478  0.13071   0.1760
## Balanced Accuracy      0.9001   0.7480   0.8663  0.74168   0.8599
```

**Note:** The above output shows the improvement related to accuracy as from 49% to 73% in the subsample test data set.    

Machine Learning Algorithm - Random Forest:
===============================================

Random Forests works as grows many classification trees. To classify a new object from an input vector, random forest put the input vector down each of the trees in the forest. Each tree gives a classification, and we say the tree "votes" for that class. The forest chooses the classification having the most votes (over all the trees in the forest).

```r
set.seed(123)
rf_model<- randomForest(classe~.,data=training, importance=T)
pred <- predict(rf_model, testing, type = "class")
confusionMatrix(pred, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1394    1    0    0    0
##          B    1  946    8    0    0
##          C    0    2  846    9    0
##          D    0    0    1  793    1
##          E    0    0    0    2  900
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9949          
##                  95% CI : (0.9925, 0.9967)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9936          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9993   0.9968   0.9895   0.9863   0.9989
## Specificity            0.9997   0.9977   0.9973   0.9995   0.9995
## Pos Pred Value         0.9993   0.9906   0.9872   0.9975   0.9978
## Neg Pred Value         0.9997   0.9992   0.9978   0.9973   0.9998
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2843   0.1929   0.1725   0.1617   0.1835
## Detection Prevalence   0.2845   0.1947   0.1748   0.1621   0.1839
## Balanced Accuracy      0.9995   0.9973   0.9934   0.9929   0.9992
```

**Note:** Finally random forest classified all output variable label very well with great accuracy (99%). It's good accuracy from last model (Decision tree) and we are expected this accuracy. 

**Applying Random forest model on in-sample Test data :**
========================================================

We will used random forest model on in-sample test data (new_data_test) first time. We use the following formula, which yielded a much better prediction in in-sample:
Pml_write_files function for generate files with predictions to submit for assignment. 

```r
rf_pred <-  predict(rf_model, new_data_test, type = "class")

pml_write_files <-  function(x){
        n = length(x)
        for(i in 1:n){
                filename = paste0("problem_id_",i,".txt")
                write.table(x[i],file=filename,quote=FALSE,
                            row.names=FALSE,col.names=FALSE)
        }
}

pml_write_files(rf_pred)
```













   
   
   
   
   
   
   
