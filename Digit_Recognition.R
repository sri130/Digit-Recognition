##load required libraries
library(caTools)
library(caret)
library(kernlab)
library(dplyr)
library(readr)
library(ggplot2)
library(gridExtra)

## set the working directory
setwd("H:/Srilekha/PGDDS/pred analysis 2/SVM/Assignment")


## Read the train csv file given
DigitRecog<-read.csv("mnist_train.csv",header=F, stringsAsFactors = F)
View(DigitRecog)

## check for nulls

sum(is.na(DigitRecog))

## Since the data set is huge, taking a random sample of it as training data.
set.seed(20)
trainindices<-sample.split(DigitRecog$V1,SplitRatio = 0.1)
digitTrain<-DigitRecog[trainindices,]

# Read the test csv file given and check for nulls.
DigitRecog_test<-read.csv("mnist_test.csv",header=F, stringsAsFactors = F)

sum(is.na(DigitRecog_test))

### Since the data set is huge, taking a random sample of it as test data.
set.seed(60)

testindices1<-sample.split(DigitRecog_test$V1,SplitRatio = 0.1)

digitTest2<-DigitRecog_test[testindices1,]


## Checking the sampled training data for nulls,

View(digitTrain)

sapply(digitTrain, function(x) sum(is.na(x)))

sum(is.na(digitTrain))

## checking the dimensions, structure, summary and first few rows of the training data.

dim(digitTrain)

str(digitTrain)

summary(digitTrain)

head(digitTrain)

##by observation first column contains all digits from 0-9, so taking this as the target variable which has to be classified.

## Converting our target class type in training and test data sets to factor.

digitTrain$V1<-as.factor(digitTrain$V1)

digitTest2$V1<-as.factor(digitTest2$V1)

## confirming it got changed.

str(digitTrain$V1)

str(digitTest2$V1)

##Scaling data
# Scaling equalises the influence of all the variables as they are in the same range.
# we should scale all the variables except our target variable
## Since all other values are in pixels and maximum pixel value is 255, dividing by 255 scales all variables.

digitTrain[,-c(1)]<-as.data.frame(digitTrain[,-c(1)]/255)

digitTest2[,-c(1)]<-as.data.frame(digitTest2[,-c(1)]/255)

## Model Building and testing the built model on the test data set###

## Let's first build a linear model and check the accuracy.
model_linear<-ksvm(V1~.,data = digitTrain,scale=FALSE, kernel="vanilladot")

## Predicting the linear model against the test data set
predicted_linear<-predict(model_linear,digitTest2)

## executing confusion matrix to see how the model behaves on the test data and check it's accuracy.
confusionMatrix(predicted_linear,digitTest2$V1)


#Overall Statistics of a linear model.

#Accuracy :  0.926         
#95% CI : (0.908, 0.9415)
#No Information Rate : 0.114           
#P-Value [Acc > NIR] : < 2.2e-16       
#Kappa : 0.9177          
#Mcnemar's Test P-Value : NA     


## 92.6% is not so good, so building an RBF model

Digit_model_rbf<-ksvm(V1~.,data = digitTrain, scale= FALSE, kernel="rbfdot")

## Predicting the RBF model against the test data set
predicted_rbf<-predict(Digit_model_rbf,digitTest2)

## ## executing confusion matrix to see how the model behaves on the test data and check it's accuracy.
confusionMatrix(predicted_rbf,digitTest2$V1)

#Overall Statistics of RBF model.

# Accuracy :  0.956            
#   95% CI : (0.9414, 0.9679)
#    No Information Rate : 0.114           
#    P-Value [Acc > NIR] : < 2.2e-16                                                
#   Kappa :  0.9511          
#   Mcnemar's Test P-Value : NA                 

# The sensitivy, specificity and balanced accuracy values are much higher in this case compared to linear model's confusion matrix.


### Hyperparameter tuning and Cross Validation 

# We will use the train function from caret package to perform Cross Validation. 

#traincontrol function Controls the computational nuances of the train function.
# i.e. method =  CV means  Cross Validation.
#      Number = 2 implies Number of folds in CV.

trainControl <- trainControl(method="cv", number=5)

# Metric <- "Accuracy" implies our Evaluation metric is Accuracy.

metric <- "Accuracy"

#Expand.grid functions takes set of hyperparameters, that we shall pass to our model.
set.seed(7)
grid <- expand.grid(.sigma=c(0.01,0.025,0.05), .C=c(1,3,5))

#train function takes Target ~ Prediction, Data, Method = Algorithm
#Metric = Type of metric, tuneGrid = Grid of Parameters,
# trcontrol = Our traincontrol method.

fit.svm <- train(V1~., data=digitTrain, method="svmRadial", metric=metric, 
                 tuneGrid=grid, trControl=trainControl)  

print(fit.svm)

plot(fit.svm)
#--------------------------------------END---------------------------------------------------------------##