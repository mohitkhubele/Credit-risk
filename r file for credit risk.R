germendata<-read.csv("file:///C:/Users/Mohit/Desktop/german project/after excel.csv")

str(germendata)
germendata$Y<-as.factor(germendata$Y)
str(germendata)

summary(germendata)
tail(germendata)

dim(germendata)
any(is.na(germendata))#finding missing value

#Drop the duplicate records.
germendata <- germendata[!duplicated(germendata),]

#check class inbalance
table(germendata$Y)

###split train data into train and validation split
library(caret)
set.seed(786)
train_rows <- createDataPartition(germendata$Y,p = 0.7,list = F)
Train<-germendata[train_rows,]
validation<-germendata[-train_rows,]

################# Model Building #####################
#1.logistic model
log_reg <- glm(Y ~ ., data = Train, family = "binomial")
summary(log_reg) # model is not significant 

Step1 <- stepAIC(log_reg, direction="both")

#build another model with significant variables
summary(log_reg) 

vif(log_reg) #no collinearity

#cutoff value for output probabilities

prod_train<-predict(log_reg,type = "response")
prod_train<-ifelse(prod_train>0.5,1,0)

pred <- ROCR::prediction(prod_train, Train$Y)

library(ROCR)

##glm_train_pre2<-ifelse(prod_train>0.5,1,0)

perf <- ROCR::performance(pred,  "tpr", "fpr")
perf_auc <- ROCR::performance(pred, measure="auc")
auc <- perf_auc@y.values[[1]]

print(auc) #71.42

cutoffs <- data.frame(cut= perf@alpha.values[[1]], fpr= perf@x.values[[1]], 
                      tpr=perf@y.values[[1]])

cutoffs <- cutoffs[order(cutoffs$tpr, decreasing=TRUE),]

plot(perf, colorize = TRUE, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.2,1.7))

#Predictions
predictTrain = predict(log_reg, type="response", newdata=Train)

pred_class_train <- ifelse(predictTrain > 0.5, "yes", "no")
table(Train$Y,pred_class_train) #78%

# Confusion matrix with threshold of 0.5
conf_matrix_train <- table(Train$Y, predictTrain > 0.5)
accuracy <- sum(diag(conf_matrix_train))/sum(conf_matrix_train)
print(accuracy)  #78.85

specificity <- conf_matrix_train[1, 1]/sum(conf_matrix_train[1, ])
print(specificity) #90%
sensitivity <- conf_matrix_train[2, 2]/sum(conf_matrix_train[2, ])
print(sensitivity) #58.08

# Predictions on the test set

predictTest = predict(log_reg, type="response", newdata=validation)
pred_class_test <- ifelse(predictTest > 0.5, "yes", "no")
table(validation$Y,pred_class_test)

# Confusion matrix with threshold of 0.5
conf_matrix_test <- table(validation$Y, predictTest > 0.5)
accuracy <- sum(diag(conf_matrix_test))/sum(conf_matrix_test)
print(accuracy) # 77.66

specificity <- conf_matrix_test[1, 1]/sum(conf_matrix_test[1, ])
print(specificity) #86.66
sensitivity <- conf_matrix_test[2,2]/sum(conf_matrix_test[2, ])
print(sensitivity) #56.66

############################ Naive Bayes ###########################
library(e1071)
model_nb<-naiveBayes(Train$Y ~ . ,Train)
#Response of the model
model_nb

#Predict the  Status on the validation data 
pred_T <- predict(model_nb,Train)
pred <- predict(model_nb,validation)
table(pred)

#Confusion Matrix
library(caret)
confusionMatrix(pred, validation$Y) # acc = 77, sen = 84, spec = 0.58
confusionMatrix(pred_T,Train$Y) # 77, sen = 84, specificity =84

###################k fold cross validation with cart ################
library(caret)
ctrl <- trainControl(method = "repeatedcv", number = 5, savePredictions = TRUE,repeats = 4)
# train the model 
model_rpart<- train(Y~., data = germendata, trControl=ctrl, method="rpart")

# make predictions
pred_train<- predict(model_rpart,Train)
pred_test <- predict(model_rpart,validation)

# summarize results
confusionMatrix(pred_train,Train$Y) #acc = 73 sen = 85  spec = 43
confusionMatrix(pred_test,validation$Y) #acc = 77.33 sen = 87 spec = 54

############################### Decision Tree #############################
#Decision tree with party
library(party)
mytree <- ctree(Y~., Train)
print(mytree)
plot(mytree,type="simple")


#Misclassification error for Train
tab<-table(predict(mytree), Train$Y)
print(tab)
1-sum(diag(tab))/sum(tab) # error 0.2298
confusionMatrix(tab) # accu = 70 spec = 2 sens = 99.5

#Misclassification error for Test
test_pred <- predict(mytree,validation)
tab1<-table(test_pred, validation$Y)
print(tab1)
1-sum(diag(tab1))/sum(tab1) # error 0.303
confusionMatrix(tab1) # accu = 69.67 spec = 0.011 sens = 99.04
################### Random Forest ##################
library(randomForest)
set.seed(222)
rf <- randomForest(Y~., data=Train,
                   ntree = 100,
                   mtry = 2,
                   importance = TRUE,
                   proximity = TRUE)
print(rf) # class.error 0.13,0.14 oob error = 13.83
attributes(rf) 
plot(rf)

# Prediction & Confusion Matrix - train data
library(caret)
p1 <- predict(rf, Train)
confusionMatrix(p1, Train$Y) # acc = 99.95 , sen = 99.95 , spec = 99.95

# # Prediction & Confusion Matrix - validation data
p2 <- predict(rf, validation)
confusionMatrix(p2, validation$Y) # acc = 77, sen = 90 , spec = 44.44

#highly overfitting
#lets tune mtry value
# Tune mtry
t <- tuneRF(Train[,-21], Train[,21],
            stepFactor = 0.5,
            plot = TRUE,
            ntreeTry = 300,
            trace = TRUE, 
            improve = 0.05) ## OOB error = 13.26% at mtry =4

# No. of nodes for the trees
hist(treesize(rf),
     main = "No. of Nodes for the Trees",
     col = "green") # it is in between 1000 to 1100

# Variable Importance
varImpPlot(rf)
importance(rf)
varUsed(rf)

################### SVM ####################

###create dummies for factor varibales 
dummies <- dummyVars(Y~.,data=germendata)

x.train = predict(dummies, newdata = Train)
y.train = Train$Y
x.validation = predict(dummies, newdata = validation)
y.validation = validation$Y

# Building the model on train data
model  =  svm(x = x.train, y = y.train, type = "C-classification", kernel = "radial", cost = 10)
summary(model)
# Predict on train and test using the model
pred_train<-predict(model,x.train)
pred_test<-predict(model,x.validation)
# Build Confusion matrix
confusionMatrix(pred_train,y.train) # acc = 93.47 sen = 93.27 spec = 93.66
confusionMatrix(pred_test,y.validation) #acc =  sens = 82.22  spec = 81.69

############ adaboost ##############

library(ada) 
model = ada(Y ~ ., iter = 20,data = Train, loss="logistic")
model

# predict the values using model on test data sets. 
pred = predict(model, Train);
pred 
pred_test<- predict(model, validation)
pred_test

confusionMatrix(Train$Y,pred) #acc=82.7 sen = 82.69 spec = 82.71
confusionMatrix(validation$Y,pred_test) #acc=81.53 sen = 82.18 spec = 80.91
