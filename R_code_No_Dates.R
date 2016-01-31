#Stat 642
#Chuxin Xue
#Gavin McCullion
#Omkar Kulkarni
# Data set WITHOUT range variable

# Run on actual testdata
library(rattle)
library(dplyr)
library(rpart)
library(randomForest)
library(kernlab)
library(gbm)
library(rpart)
library(caret)
library(ada)
library(ggplot2)
library(data.table)
library(effects)
library(car)
library(glmnet)
library(plyr)
library(nnet)


#Importing data
setwd("/Users/ShireeXue/Desktop/Spring 2015/Data Mining/Final Project/")
rawdata<- read.csv("/Users/ShireeXue/Desktop/Spring 2015/Data Mining/Final Project/traindata.csv", header=TRUE)
summary(rawdata)
str(rawdata)

#convert variables names to lowercase
names(rawdata) <- tolower(names(rawdata))

#View class of variables, check to see if anything is misclassified
sapply(rawdata,class)

#Create new variable debt

rawdata$debt<- ifelse(rawdata$tot_open_amt== 0, 1, 0)


# Outlier Detection
#Check outliers for numeric variables. none outliers. Used 2.2 Standard deviation from Q1 and Q3
#There's no way to check outliers for categorical data

hist(rawdata$tot_invoice_amt,breaks=80)
hist(rawdata$tot_open_amt,breaks=80)
hist(rawdata$tot_paid_amt,breaks=80)
hist(rawdata$contract_fee,breaks=80)


# correlation matrix - identify highly correlated variables: 
#current-balance and tot_open_amt

mc <- cor(rawdata[which(sapply(rawdata, is.numeric))], use="complete.obs")
mc[upper.tri(mc, diag=TRUE)] <- NA
mc

# Identify variables with too many levels - may affect model performance/efficiency 
levels(rawdata$city)
levels(rawdata$zip)
levels(rawdata$state)

#Remove unwanted variables
mydata<-select(rawdata,-obs,-currentbalance,-customerid,-zip,-city,-state,-renewal_year,-renewal_month,-activated_year,-activated_month)


# #Exploratory analysis 
# Data.Table 
# Churn rate by region/ age / rate_plan/ credit 

DT<- data.table(mydata)

Region_DT<- DT[,.(Avg_Churn.Sum = mean(churn)),by=region]
Age_DT<- DT[,.(Avg_Churn.Sum = mean(churn)),by=age_range]
Rate_plan_DT<- DT[,.(Avg_Churn.Sum = mean(churn)),by=rate_plan]
Credit_DT<- DT[,.(Avg_Churn.Sum = mean(churn)),by=credit_approval]


# GGPLOT

#plot of churn rates for credit_approval variable
approvalplot <- ggplot(mydata, aes(x=credit_approval, y=churn))
approvalplot + stat_summary(fun.y="mean", geom="bar", fill="darkblue") + ggtitle("Churn Rates for Credit Approval Levels")


#plot of churn rates for debt variable
debtplot <- ggplot(myata, aes(x=debt, y=churn))
debtplot + stat_summary(fun.y="mean", geom="bar", fill="blue") + ggtitle("Churn Rates for Debt Variable")

#plot of churn rates for age_range variable
ageplot <- ggplot(myata, aes(x=age_range, y=churn))
ageplot + stat_summary(fun.y="mean", geom="bar", fill="darkblue") + ggtitle("Churn Rates for Age Levels")



# Identify interaction term (Prepare for LOGISTIC INTERACTION)
install.packages("effects")
library(effects)

mydata$churn<-as.factor(mydata$churn)

# Tot_open_amount and tot_paid_amt : Interaction effect exist
testlm1<-glm(churn ~ tot_open_amt + tot_paid_amt+tot_open_amt:tot_paid_amt,family=binomial,data=mydata)
plot(effect(term="tot_open_amt:tot_paid_amt",mod=testlm1,default.levels=5),multiline=TRUE)

# Credit Approval and Debt - No interaction effect
testlm2<-glm(churn ~ credit_approval + debt+credit_approval:debt,family=binomial,data=mydata)
plot(effect(term="credit_approval:debt",mod=testlm2,default.levels=2),multiline=TRUE)

# RatePlan and age -  Yes. there's interacction 
testlm3<-glm(churn ~ rate_plan + age_range+rate_plan:age_range,family=binomial,data=mydata)
plot(effect(term="rate_plan:age_range",mod=testlm3,default.levels=5),multiline=TRUE)

# Debt and age -  Yes. there's interacction 
# Debt customers have high churn rate until age hits 90S
testlm4<-glm(churn ~ debt+age_range+ debt:age_range,family=binomial,data=mydata)
plot(effect(term="debt:age_range",mod=testlm4,default.levels=2),multiline=TRUE)


# Original Data Partioning  70/20/10
set.seed(12345)
nall <- 371933
ntrain <- floor(0.7*nall)
ntest <- floor(0.20*nall)
nvalidate <- floor(0.10*nall)
index <- seq(1:nall)
train <- sample(index,ntrain)
newindex <- index[-train]
test <- sample(newindex,ntest)
newnewindex <- index[-c(train,test)]
validate <- newnewindex

#Original data partitioning
traindata<- mydata[train,]
testdata<- mydata[test,]
validatedata<- mydata[validate,]



# Decision Tree
library(rpart)

# Data Preparation
traintree<-traindata
testtree<-testdata
valitree<-validatedata

# Convert binary to factors as Trees work better with catgorical variable
sapply(traindata,class)


factcols <- c("churn", "debt")
traintree[,factcols] <- data.frame(apply(traintree[factcols], 2, as.factor))
testtree[,factcols] <- data.frame(apply(testtree[factcols], 2, as.factor))
valitree[,factcols] <- data.frame(apply(valitree[factcols], 2, as.factor))


## Model building 
# Cross validation=10
# Tried increase and decrease minsplit and minbucket, result stay very similar
# Grow trees and find min cp=0.01
fit <- rpart(churn ~., method="class", data=traintree)
printcp(fit)
plotcp(fit)
# Prune trees
tree<- prune(fit, minsplit=100, minbucket =round(minsplit/3),cp = fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"],xval=10)
summary(tree)


#Visualize the tree model
par(xpd = TRUE)
plot(tree, compress = TRUE)
text(tree, use.n = TRUE)

# Confusion matrix
predtree<- predict(tree,traintree, type="class")
treetable1<-table(predtree, traintree$churn)
treetable1
# Error rate for traindata
treeerror1=(treetable1[1,2]+treetable1[2,1])/260353
treeerror1
#0.226

# Model on testdata 
predtest <- predict(tree, newdata = testtree, type = c("class"))

# Confusion matrix of testdata
treetable2<-table(predtest, testtree$churn)
treetable2
# Error rate
treeerror2=(treetable2[1,2]+treetable2[2,1])/74386
treeerror2
#0.2275

# Predict on validation data
predvali <- predict(tree, newdata = valitree, type = c("class"))


# Confusion matrix of validata
treetable3<-table(predvali, valitree$churn)
treetable2
# error rate on validation 
treeerror3=(treetable3[1,2]+treetable3[2,1])/37194
treeerror3
#0.224



# Support Vector Machine

#Data preparation
# Transform categories into dummy variable
# city, state , zip , we can use PCA or cluster before dummy

age_dummy <- model.matrix(~ factor(mydata$age_range) - 1)
region_dummy<- model.matrix(~ factor(mydata$region) - 1)
credit_approval_dummy<- model.matrix(~ factor(mydata$credit_approval) - 1)
contact_method_dummy<- model.matrix(~ factor(mydata$contact_method) - 1)
rate_plan_dummy<-model.matrix(~ factor(mydata$rate_plan) - 1)


# Create continuous dataset by creating dummy varialbes
mydata.dummy<-cbind(mydata,age_dummy,region_dummy,credit_approval_dummy,contact_method_dummy,rate_plan_dummy)
mydata.dummy<-select(mydata.dummy,-age_range,-region,-credit_approval,-contact_method,-rate_plan)


dummy.train<- mydata.dummy[train,]
dummy.test<- mydata.dummy[test,]
dummy.vali<- mydata.dummy[validate,]

# Take 1% sample  from traindata
set.seed(1)
svmsam1<- sample_frac(traindata.dummy,0.01, replace = FALSE)


#Rbf Kernel
myresults3 <- matrix(nrow=100,ncol=3)
myi <- 1
for (myC in seq(0.3,1,by=0.1)){
  for (mysigma in seq(1,10,by=1)){
    svp <- ksvm(churn~., data=svmsam1,type="C-svc",kernel="rbf",kpar=list(sigma=mysigma),C=myC,scaled=c(),cross=5)
    myresults3[myi,] <- c(mysigma,myC,cross(svp))
    myi <- myi + 1
  }
}
myresults3[1:10,]
plot(myresults3[,3])

# Minimum error 0.3665027 (RBF)
which.min(myresults[,3])
myresults[17,3]


presam3 = predict(svprbf2,svmsam2[,-6])
table(svmsam2[,6],presam3)
# Compute accuracy
sum(presam3==svmsam2[,6])/length(svmsam2[,6])
svmsam1


presam2 = predict(svprbf,svmsam1[,-6])
table(svmsam1[,6],presam2)
# Compute accuracy
sum(presam2==svmsam1[,6])/length(svmsam1[,6])


#Poly Kernel -BEST WHEN C=0.6

resultpoly <- matrix(nrow=100,ncol=2)
myi <- 1
for (myC in seq(1,100,by=5)){
  svp <- ksvm(churn~., data=svmsam1,type="C-svc",kernel="poly",C=myC,scaled=c(),cross=5)
  resultpoly[myi,] <- c(myC,cross(svp))
  myi <- myi + 1
}

resultpoly[1:10,]
plot(myresults3[,3])
which.min(resultpoly[,2])
resultpoly[4,2]
resultpoly[1:5,]
# 0.6 0.3318091 error LOWEST


newsvppoly <- ksvm(as.matrix(svmsam1[,-6]),as.matrix(svmsam1[,6]),type="C-svc",kernel="poly",C=0.6,scaled=c(),cross=5)
# Cross validatio error 0.229
#0.7688172


svp <- ksvm(as.matrix(svmsam1[,-6]),as.matrix(svmsam1[,6]),type="C-svc",kernel="poly",C=0.5,scaled=c(),cross=5)
svp
#0.347388
svp <- ksvm(as.matrix(svmsam1[,-6]),as.matrix(svmsam1[,6]),type="C-svc",kernel="poly",C=1,scaled=c(),cross=5)
svp
#0.346075
svp <- ksvm(as.matrix(svmsam1[,-6]),as.matrix(svmsam1[,6]),type="C-svc",kernel="poly",C=0.8,scaled=c(),cross=5)
svp
#0.407898
svp <- ksvm(as.matrix(svmsam1[,-6]),as.matrix(svmsam1[,6]),type="C-svc",kernel="poly",C=10,scaled=c(),cross=5)
svp
#0.3557
svp <- ksvm(as.matrix(svmsam1[,-6]),as.matrix(svmsam1[,6]),type="C-svc",kernel="poly",C=50,scaled=c(),cross=5)
#0.38208
svp <- ksvm(as.matrix(svmsam1[,-6]),as.matrix(svmsam1[,6]),type="C-svc",kernel="poly",C=300,scaled=c(),cross=5)
#0.354646
svp <- ksvm(as.matrix(svmsam1[,-6]),as.matrix(svmsam1[,6]),type="C-svc",kernel="poly",C=600,scaled=c(),cross=5)
svp
#0.358699
1     



# Linear 
resultlinear <- matrix(nrow=100,ncol=2)
myi <- 1
for (myC in seq(1,100,by=5)){
  svp <- ksvm(churn~., data=svmsam1,type="C-svc",kernel="vanilladot",C=myC,scaled=c(),cross=5)
  resultlinear[myi,] <- c(myC,cross(svp))
  myi <- myi + 1
}
plot(resultlinear[,2])
which.min(resultlinear[,2])
resultlinear[1,]

# Linear Best C=1 cross validation error= 0.2266762

newsvplinear <- ksvm(as.matrix(svmsam1[,-6]),as.matrix(svmsam1[,6]),type="C-svc",kernel="vanilladot",C=1,scaled=c(),cross=5)
newsvplinear  
# Cross validatio error 0.22533



set.seed(2)
svmsam2<- sample_frac(dummy.test,0.01, replace = FALSE)


presam1 = predict(newsvplinear,svmsam2[,-6])
table(svmsam2[,6],presam1)
# Compute accuracy
sum(presam1==svmsam2[,6])/length(svmsam2[,6])
#0.756

## Testdata sample 2
set.seed(3)
svmsam3<- sample_frac(dummy.test,0.01, replace = FALSE)

#Predict 
presam2 = predict(newsvplinear,svmsam3[,-6])
table(svmsam3[,6],presam2)
# Compute accuracy
sum(presam2==svmsam3[,6])/length(svmsam3[,6])
#0.7634

## Testdata sample 3
set.seed(4)
svmsam4<- sample_frac(dummy.test,0.01, replace = FALSE)

#Predict 
presam3 = predict(newsvplinear,svmsam4[,-6])
table(svmsam4[,6],presam3)
# Compute accuracy
sum(presam3==svmsam4[,6])/length(svmsam4[,6])
#0.7715

## Testdata sample 4
set.seed(5)
svmsam5<- sample_frac(dummy.test,0.01, replace = FALSE)

#Predict 
presam4 = predict(newsvplinear,svmsam5[,-6])
table(svmsam5[,6],presam4)
# Compute accuracy
sum(presam4==svmsam5[,6])/length(svmsam5[,6])
#0.74




#Random Forest

#Data preparation
#Drop city, zip , state1
library(randomForest)

#Convert binary to factors
lapply(traindata,class)

traindata$churn<-as.factor(traindata$churn)
traindata$debt<-as.factor(traindata$debt)
testdata$churn<-as.factor(testdata$churn)
testdata$debt<-as.factor(testdata$debt)

set.seed(1)
# Parameter search
tuneRF(traindata[,c(1:10,12)],traindata$churn,ntreeTry=100,stepFactor=1.5,improve=0.01,plot=TRUE)
#Lowest mtry=3, ntree=100
tuneRF(traindata[,c(1:10,12)],traindata$churn,ntreeTry=500,stepFactor=1.5,improve=0.01,plot=TRUE)
# Lowest mtry=3 ntree=500

myrf <- randomForest(churn ~.,traindata,ntree=500, mtry=3, importance=TRUE)
myrf
# OOB error 17.01% - Best model 

myrf2 <- randomForest(churn ~.,traindata,ntree=100, mtry=3, importance=TRUE)
myrf2
# OOB error 17.19%


#Important variables
myrf$importance
varImpPlot(myrf)


#Testdata 500-tree 
rfpred2 <- predict(myrf,newdata=testdata,type="response")

rftable2<-table(rfpred2,testdata$churn)
rftable2
# Error rate 
rferror2=(rftable2[1,2]+rftable2[2,1])/74386
rferror2


# Test data 100-tree
rfpred4 <- predict(myrf2,newdata=testdata,type="response")
rftable4<-table(rfpred4,testdata$churn)
rftable4
# Error rate 
rferror4=(rftable4[1,2]+rftable4[2,1])/74386
rferror4


#Validatedata 500-tree
rfpred3 <- predict(myrf,newdata=validatedata,type="response")

rftable3<-table(rfpred3,validatedata$churn)
rftable3
# Error rate for validate data
rferror3=(rftable3[1,2]+rftable3[2,1])/37194
rferror3



# ROC Cruve for our best model in Random Forest
ROCrfpre<- predict(myrf,testdata[,-11],type='prob')
plot(performance(prediction(ROCrfpre[,2],testdata$churn), 'tpr', 'fpr'))


## Boosting

#convert binary to factors
traindata$churn<-as.factor(traindata$churn)
traindata$debt<-as.factor(traindata$debt)
testdata$churn<-as.factor(testdata$churn)
testdata$debt<-as.factor(testdata$debt)
validatedata$churn<-as.factor(validatedata$churn)
validatedata$debt<-as.factor(validatedata$debt)

#GMB Packages

library(gbm)
boost.train_gbm<-gbm(churn~., data=traindata,distribution="adaboost", n.trees=150,shrinkage=0.1, interaction.depth=3)
summary(boost.train_gbm)
boost.train_gbm

# Relative influence show that tot_open_amt, credit_approval, rate_plan, tot_invoice_amt 
varImp(boost.train_gbm)
plot(boost.train_gbm)

# ADA package
#Train model to find best parameters

library(caret)


# Caret Train model
library(caret)
objControl<-trainControl(method='cv', number=5, returnResamp='none', 
                         summaryFunction=twoClassSummary,classProbs=TRUE)
objModel<-train(traindata[,c(1:10,12)],traindata[,11],
                method='ada',
                trControl=objControl,
                metric="ROC")

summary(objModel)
# Optimal Result: iter=150 nu=0.1

library(ada)
boost.train<-ada(churn~., data=traindata,loss="ada",type="discrete", iter=150, nu=0.1)
boost.train
summary(boost.train)
# Out-of-bag error 19.6%

# Prediction on Traindata
boost.train.predict<-predict(boost.train,traindata,type="vector")

boost.table1<-table(boost.train.predict, traindata$churn)
boost.table1

# Error rate for traindata
boost.error1=(boost.table1[1,2]+boost.table1[2,1])/260353
#Accuracy
1-boost.error1
# Train data accuracy 80.6%


# Prediction on Testdata
boost.test.predict<-predict(boost.train,testdata,type="vector")

boost.table2<-table(boost.test.predict, testdata$churn)
boost.table2

# Error rate for testdata
boost.error2=(boost.table2[1,2]+boost.table2[2,1])/74386
#Accuracy
1-boost.error2
# Test data accuracy : 80.4%


# Prediction on Testdata
boost.vali.predict<-predict(boost.train,validatedata,type="vector")

boost.table3<-table(boost.vali.predict, validatedata$churn)
boost.table3

# Error rate for testdata
boost.error3=(boost.table3[1,2]+boost.table3[2,1])/37194
#Accuracy
boost.error3
1-boost.error3

# ROC Curve for our best model in Boosting
ROCboostpre<- predict(boost.train,testdata,type='prob')
ROCboostpre
plot(performance(prediction(ROCboostpre[,2],testdata$churn), 'tpr', 'fpr'))


#logistic regression

logfit <- glm(churn~., family=binomial(link=logit), data=mydata[train,])
logfitstep <- step(logfit)
logfit <- glm(churn ~ contract_fee + region + tot_open_amt + age_range +
                tot_invoice_amt + num_invoices + contact_method + credit_approval + rate_plan + debt + rate_plan:age_range, debt:age_range,
              data=mydata[train,])
summary(logfit)

#VIF
vif(logfit)

#rate_plan and contact_method are above 4

#predict
logfit.predict <- ifelse(predict(logfit, mydata[test,], type="response") > 0.5, TRUE, FALSE)
y <- mydata[test,]
confusion <- table(logfit.predict, as.logical(y$churn))
confusion  <- cbind(confusion, c(1 - confusion[1,1]/(confusion[1,1]+confusion[2,1]), 1 - confusion[2,2]/(confusion[2,2]+confusion[1,2])))
confusion  <- as.data.frame(confusion)
names(confusion) <- c('FALSE', 'TRUE', 'class.error')
confusion
log1error=(confusion[1,2]+confusion[2,1])/74386
log1error
#error rate - 0.216

#glmnet version
install.packages("glmnet")
library(glmnet)

#glmnet only works on matrices - need to transform using model.matrix

rawdatamatrix <- model.matrix(churn ~ region + contract_fee + tot_open_amt + tot_invoice_amt + tot_paid_amt + num_invoices
                              + age_range + credit_approval + contact_method + rate_plan + debt, data=mydata)

#drop intercept from model matrix, and define x and y variables
x <- rawdatamatrix[,-1]
x1 <- x[train,]
x2 <- x[test,]
y1 <- mydata[train,]
y2 <- mydata[test,]
library(glmnet)
fit1 <-glmnet(x1, y1$churn, family="binomial")
print(fit1)
plot(fit1)
coef(fit1, s=0.01)

log1pred <- predict(fit1, x, s=0.005, type="class")

logtable <- table(log1pred, y$churn)
logtable
log1error=(logtable[1,2]+logtable[2,1])/260353
log1error
#0.216 error rate

#now to try with cross validation:

log2 <- cv.glmnet(x, y$churn, family="binomial")
plot(log2)

log2$lambda.min
#0.00027
coef(log2, s = "lambda.min")

#Predict on train
log2pred <- predict(log2, x1, s="lambda.min", type="class")
log2tables <- table(log2pred, y1$churn)
log2tables
log2error=(log2tables[1,2]+log2tables[2,1])/260353
log2error
#0.2116 error rate


#Predict on test
log2pred2 <- predict(log2, x2, s="lambda.min", type="class")
log2tables2 <- table(log2pred2, y2$churn)
log2tables2
log2error2=(log2tables2[1,2]+log2tables2[2,1])/74386
log2error2
#0.213 error rate


#neural network

#data prep
install.packages("plyr")
library(plyr)

nndata<- mydata
for(level in unique(nndata$region)){
  nndata[paste("dummy", level, sep = "_")] <- ifelse(nndata$region == level, 1, 0)
}

nndata <- mutate(nndata, region1 = dummy_E_S_CENTRAL + dummy_S_ATLANTIC)
nndata <- mutate(nndata, region2 = dummy_W_S_CENTRAL + dummy_E_N_CENTRAL)
nndata <- mutate(nndata, region3 = dummy_AK + dummy_HI + dummy_NEW_ENGLAND + dummy_W_N_CENTRAL + dummy_MOUNTAIN + dummy_PACIFIC + dummy_MID_ATLANTIC)

nndata <- nndata[,-c(13:23)]

rawdatamatrix <- model.matrix(churn~ contract_fee + tot_open_amt + tot_invoice_amt + tot_paid_amt + num_invoices
                              + age_range + credit_approval + contact_method + rate_plan + debt + region1 + region2 + region3,
                              data=nndata)

nndata2 <- rawdatamatrix[,-1]
nndata2 <- as.data.frame(nndata2)
nndata2 <- scale(nndata2)
nndata2 <- cbind(nndata2, mydata$churn)
nndata2 <- as.data.frame(nndata2)
detach("package:dplyr", unload=TRUE)
library(plyr)
nndata2 <- rename(nndata2, c("V36" = "churn"))

library(nnet)
set.seed(3)
nn1 <- nnet(churn~.,data=nndata2[train,], entropy=T,size=3,decay=0,maxit=2000,trace=T)

yhat <- (predict(nn1, newdata = nndata2[test,], type="class"))
nntable1 <- table(yhat, nndata[test,]$churn)
nntable1
nn1error=(nntable1[1,2]+nntable1[2,1])/74386
nn1error

#error rates for neural net sizes:
#size 1: 0.210
#size 2: 0.207
#size 3: 0.202
#size 4: 0.204
#size 5: 0.1997
#size 6: 0.1996
#size 7: 0.197
#size 8: 0.195
#size 9: 0.193
#size 10: 0.195

library(e1071)
tmodel <- tune.nnet(churn~., data=nndata2[train,], size = 1:5)
summary(tmodel)
plot(tmodel)
tmodel$best.model




# Best Model to Run on FINAL Test data

rawtest<- read.csv("/Users/ShireeXue/Desktop/Spring 2015/Data Mining/Final Project/testdata.csv", header=TRUE)
#convert variables names to lowercase
names(rawtest) <- tolower(names(rawtest))

#Create new variable debt
library(dplyr)

rawtest$debt<- ifelse(rawtest$tot_open_amt== 0, 1, 0)
finaldata<-select(rawtest,-currentbalance,-customerid,-zip,-city,-state,-renewal_year,-renewal_month,-activated_year,-activated_month)
finaldata$debt<-as.factor(finaldata$debt)

#Validatedata 500-tree
finalrf <- predict(myrf,newdata=finaldata,type="response")


# Evaluation 

# ROC Curve
# ROC Cruve for our best models
library(ROCR)
testdata$debt<- as.numeric(testdata$debt)
ROCboostpre<- predict(boost.train,testdata,type='prob')
ROCrfpre<- predict(myrf,testdata[,-11],type='prob')
ROCnn <- predict(nn1, newdata = nndata2[test,], type="raw")

boostperf<-performance(prediction(ROCboostpre[,2],testdata$churn), 'tpr', 'fpr')
rfperf<- performance(prediction(ROCrfpre[,2],testdata$churn), 'tpr', 'fpr')
NNperf<-  performance(prediction(ROCnn[,1],testdata$churn), 'tpr', 'fpr')

plot(NNperf, col='green')
plot(rfperf, add = TRUE, col = 'red')
plot(boostperf, add = TRUE, col='blue')

legend("bottomright", inset=.05, title="Algorithm",
       c("NeuralNet","RandomForest","Boosting"), fill=c("green","red","blue"), horiz=TRUE)



