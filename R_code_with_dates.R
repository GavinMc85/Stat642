#Stat 642
#Chuxin Xue
#Gavin McCullion
#Omkar Kulkarni
# Data set WITH range variable

library(dplyr)
library(rpart)
library(randomForest)
library(kernlab)
library(gbm)
library(rpart)
library(ada)

#Importing data
setwd("/Users/ShireeXue/Desktop/Spring 2015/Data Mining/Final Project/")
rawdata<- read.csv("/Users/ShireeXue/Desktop/Spring 2015/Data Mining/Final Project/traindata.csv", header=TRUE)
summary(rawdata)
str(rawdata)

#convert variables names to lowercase
names(rawdata) <- tolower(names(rawdata))

#View class of variables, check to see if anything is misclassified
sapply(rawdata,class)

# Convert months to months in number
mo2Num <- function(x) match(tolower(x), tolower(month.abb))

#Format date variable for customer lifetime calculation 
rawdata<-mutate(rawdata,renewal_month=mo2Num(renewal_month),activated_month=mo2Num(activated_month),dummday="01")
rawdata<- mutate(rawdata, renewal_date=paste(renewal_month,dummday,renewal_year,sep="/"),activated_date=paste(activated_month,dummday,activated_year,sep="/"))
rawdata$renewal_num <- strptime(rawdata$renewal_date, "%m/%d/%Y")
rawdata$activated_num <- strptime(rawdata$activated_date, "%m/%d/%Y")

#Convert year and month from number to factors
rawdata$activated_year<- as.factor(rawdata$activated_year)
rawdata$activated_month<- as.factor(rawdata$activated_month)
rawdata$renewal_year<- as.factor(rawdata$renewal_year)
rawdata$renewal_month<- as.factor(rawdata$renewal_month)
rawdata$renewal_num<- as.Date(rawdata$renewal_num)
rawdata$activated_num<- as.Date(rawdata$activated_num)


# Calculate the customer lifeteime, unit in weeks. 
rawdata<-mutate(rawdata,custlife=difftime(rawdata$renewal_num,rawdata$activated_num,units="weeks"))
rawdata$custlife<- as.numeric(rawdata$custlife)
#Create new variable "loyalcust", 1 is loyal customers who renewed ,0 is not.
rawdata$loyalcust<- ifelse(rawdata$custlife== 0, 0, 1)
rawdata$debt<- ifelse(rawdata$tot_open_amt== 0, 1, 0)

#Remove redundant variables
mydata<-select(rawdata,-obs,-currentbalance,-customerid,-dummday,-zip,-city,-state,-renewal_num,-activated_num)

# Original Data Partioning 
set.seed(12345)
nall <- 371933
ntrain <- floor(0.7*nall)
ntest <- floor(0.15*nall)
nvalidate <- floor(0.15*nall)
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
# Convert binary to factors as Trees work better with catgorical variable

traintree<-select(traindata)
testtree<-select(testdata)
valitree<-select(validata)

factcols <- c("churn", "renewal_num","activated_num","loyalcust","debt")
traintree[,factcols] <- data.frame(apply(traintree[factcols], 2, as.factor))
testtree[,factcols] <- data.frame(apply(testtree[factcols], 2, as.factor))
valitree[,factcols] <- data.frame(apply(valitree[factcols], 2, as.factor))


# Model building 
# Cross validation=10
treemodel<- rpart(formula(churn ~ .), data=traintree
                  method="class",control=rpart.control(minsplit =100, minbucket = 30, cp = 0.01,maxcompete = 4, 
                                                       maxsurrogate = 5, usesurrogate = 2, xval = 10, surrogatestyle = 0, maxdepth = 30), 
                  parms=list(split="information"))
treemodel

printcp(treemodel)
plotcp(treemodel)
treemodel

#Visualize the tree model
par(xpd = TRUE)
plot(treemodel, compress = TRUE)
text(treemodel, use.n = TRUE)

# Confusion matrix
predtree<- predict(treemodel,newtrain, type="class")
treetable1<-table(predtree, newtrain$churn)
treetable1
# Error rate for traindata
treeerror1=(treetable1[1,2]+treetable1[2,1])/260353
treeerror1



# Model on testdata 
predtest <- predict(treemodel, newdata = testtree, type = c("class"))

# Confusion matrix of testdata
treetable2<-table(predtest, testtree$churn)
treetable2

treeerror2=(treetable2[1,2]+treetable2[2,1])/74386
treeerror2

#Cross-Validataion
#Grow Tree
fit <- rpart(churn ~., method="class", data=traintree)
#Prune Trees
pfit<- prune(fit, cp=   fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"])



# Boosting 
# Remove variables that have too many levels as boostging couldn't handle vars with more then 1024 levels

traindata.boost<-select(traindata,-city,-zip)
testdata.boost<-select(testdata,-city,-zip)
validata.boost<-select(validata,-city,-zip)
traindata.boost$churn<- as.factor(traindata.boost$churn)

# OPTIONAL : Convert all binary var to factor
factcols <- c("churn", "renewal_num","activated_num","loyalcust","debt")
traindata.boost[,factcols] <- data.frame(apply(traindata.boost[factcols], 2, as.factor))
testdata.boost[,factcols] <- data.frame(apply(testdata.boost[factcols], 2, as.factor))
validata.boost[,factcols] <- data.frame(apply(validata.boost[factcols], 2, as.factor))

# Find out what type of calculation the package does
library(caret)
getModelInfo()$ada$type
getModelInfo()$gbm$type


library(gbm)
boost.train2<-gbm(churn~., data=traindata.boost,distribution="adaboost", n.trees=500,shrinkage=0.01, interaction.depth=4)
summary(boost.train2)
# Relative influence show that loyal customer , custlifetime, renewal_num, num_invoices, total_open_amt have the largest influence on the model

# ADA package
#Train model to find best parameters
install.packages("pROC")
summary(objModel)
objControl<-trainControl(method='cv', number=3, returnResamp='none', 
                         summaryFunction=twoClassSummary,classProbs=TRUE)
objModel<-train(traindata.boost,traindata.boost[,17],
                method='ada',
                trControl=objControl,
                metric="ROC")
objModel

# Result: iter=50 nu=0.1

boost.train<-ada(churn~., data=traindata.boost,loss="ada",type="discrete", iter=50, nu=0.1)
#nu shrinkage parameter is 1, we can update it and try again like 0.01
summary(boost.train)



#Confusion matrix on traindata
boost.train$confusion
boost.train$fit
# 96.

# Prediction on Traindata
boost.train.predict<-predict(boost.train,traindata.boost,type="vector")

boost.table1<-table(boost.train.predict, traindata.boost$churn)
boost.table1

# Error rate for traindata
boost.error1=(boost.table1[1,2]+boost.table1[2,1])/260353
#Accuracy
1-boost.error1



# Prediction on Testdata
boost.test.predict<-predict(boost.train,testdata.boost,type="vector")

boost.table2<-table(boost.test.predict, testdata.boost$churn)
boost.table2

# Error rate for testdata
boost.error2=(boost.table2[1,2]+boost.table2[2,1])/55789
#Accuracy
1-boost.error2


# Decision Tree
library(rpart)

# Data Preparation
# Convert binary to factors as Trees work better with catgorical variable

traintree<-select(traindata)
testtree<-select(testdata)
valitree<-select(validata)

factcols <- c("churn", "renewal_num","activated_num","loyalcust","debt")
traintree[,factcols] <- data.frame(apply(traintree[factcols], 2, as.factor))
testtree[,factcols] <- data.frame(apply(testtree[factcols], 2, as.factor))
valitree[,factcols] <- data.frame(apply(valitree[factcols], 2, as.factor))


# Model building 
# Cross validation=10
treemodel<- rpart(formula(churn ~ .), data=traintree
                  method="class",control=rpart.control(minsplit =100, minbucket = 30, cp = 0.01,maxcompete = 4, 
                                                       maxsurrogate = 5, usesurrogate = 2, xval = 10, surrogatestyle = 0, maxdepth = 30), 
                  parms=list(split="information"))
treemodel

printcp(treemodel)
plotcp(treemodel)
treemodel

#Visualize the tree model
par(xpd = TRUE)
plot(treemodel, compress = TRUE)
text(treemodel, use.n = TRUE)

# Confusion matrix
predtree<- predict(treemodel,newtrain, type="class")
treetable1<-table(predtree, newtrain$churn)
treetable1
# Error rate for traindata
treeerror1=(treetable1[1,2]+treetable1[2,1])/260353
treeerror1



# Model on testdata 
predtest <- predict(treemodel, newdata = testtree, type = c("class"))

# Confusion matrix of testdata
treetable2<-table(predtest, testtree$churn)
treetable2

treeerror2=(treetable2[1,2]+treetable2[2,1])/55789
treeerror2

#Cross-Validataion
#Grow Tree
fit <- rpart(churn ~., method="class", data=traintree)
#Prune Trees
pfit<- prune(fit, cp=   fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"])



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
dummy.mydata<-cbind(mydata,age_dummy,region_dummy,credit_approval_dummy,contact_method_dummy,rate_plan_dummy)
dummy.mydata<-select(dummy.mydata,-age_range,-region,-credit_approval,-contact_method,-rate_plan)

# Continuos data with dummy variables partitioning
dummy.traindata<- dummy.mydata[train,]
dummy.testdata<- dummy.mydata[test,]
dummy.validatedata<- dummy.mydata[validate,]

sapply(traindata.boost,class)

library(kernlab)
#Poly Kernel
svp <- ksvm(as.matrix(dummy.traindata.svm[,-10]),as.matrix(dummy.traindata.svm[,10]),type="C-svc",kernel="poly",C=1,scaled=c())

#Rbf Kernel # Better accuracy
svprbf <- ksvm(as.matrix(dummy.traindata.svm[,-10]),as.matrix(dummy.traindata.svm[,10]),type="C-svc",kernel="rbf",kpar=list(sigma=1),C=1,scaled=c(),cross=5)

svp
svprbf
summary(svp)
summary(svprbf)

# On Train data
# Confusion matrix
predsvmrbf<- predict(svprbf,svmsam1[,-6])
svmrbftable1<-table(predsvmrbf,svmsam1$churn)
svmrbftable1
# Error rate for traindata
svmrbferror1=(svmrbftable1[1,2]+svmrbftable1[2,1])/260353
svmrbferror1

# On Testdata
# Confusion matrix
predsvm<- predict(svprbf,dummy.testdata.svm[,-10])
svmtable2<-table(predsvm, dummy.testdata.svm$churn)
svmtable2
# Error rate for traindata
svmerror2=(svmtable1[1,2]+svmtable1[2,1])/55789
svmerror2


#Random Forest

#Data preparation
#Drop city, zip , state1
library(randomForest)
traindata<-mutate(traindata,churn=as.factor(churn),debt=as.factor(debt),loyalcust=as.factor(loyalcust),renewal_date=as.factor(renewal_date), activated_date=as.factor(activated_date))

myrf <- randomForest(churn~.,data=traindata,importance=TRUE)
myrf

myrf$importance
varImpPlot(myrf)

#Predict on traindata
rfpred <- predict(myrf,traindata,type="response")
rftable1<-table(rfpred,traindata$churn)

# Error rate for traindata
rferror1=(rftable1[1,2]+rftable1[2,1])/260353
rferror1

# Predict on testdata
rfpred2 <- predict(myrf,newdata=testdata3,type="response")
rftable2<-table(rfpred2,testdata3$churn)
rftable2
# Error rate for testdata
rferror2=(rftable2[1,2]+rftable2[2,1])/74386
rferror2


#Predict on validate
rfpred3 <- predict(myrf,validatedata,type="response")
rftable3<-table(rfpred3,validate$churn)

# Neural Network Data prepration 
#code dummy variables for credit approval
detach("package:dplyr", unload=TRUE)
library(plyr)
rawdata <- mutate(rawdata, cahigh = ifelse(credit_approval == " 2 Approved Services", 1, 0))

for(level in unique(rawdata$credit_approval)){
  rawdata[paste("dummy", level, sep = "_")] <- ifelse(rawdata$credit_approval == level, 1, 0)
}

summary(rawdata)
rawdata <- rename(rawdata, c("dummy_ 1 Approved Services" = "dummy_1_Approved_Services", "dummy_ Manual Review" = "dummy_Manual_Review",
                             "dummy_ 3 Approved Services" = "dummy_3_Approved_Services", "dummy_ 5 Approved Services" = "dummy_5_Approved_Services",
                             "dummy_10 Approved Service" = "dummy_10_Approved_Service"))
rawdata <- mutate(rawdata, camid = dummy_1_Approved_Services + dummy_5_Approved_Services + dummy_Manual_Review + dummy_OTHER)

rawdata <- mutate(rawdata, calow = dummy_3_Approved_Services + dummy_OTHER)

rawdata <- rawdata[, -c(26:33)]

#code dummy variables for region
for(level in unique(rawdata$region)){
  rawdata[paste("dummy", level, sep = "_")] <- ifelse(rawdata$region == level, 1, 0)
}

rawdata <- mutate(rawdata, region1 = dummy_E_S_CENTRAL + dummy_S_ATLANTIC)
rawdata <- mutate(rawdata, region2 = dummy_W_S_CENTRAL + dummy_E_N_CENTRAL)
rawdata <- mutate(rawdata, region3 = dummy_AK + dummy_HI + dummy_NEW_ENGLAND + dummy_W_N_CENTRAL + dummy_MOUNTAIN + dummy_PACIFIC + dummy_MID_ATLANTIC)

rawdata <- rawdata[, -c(28:38)]

#code dummy variables for contact_method
rawdata <- mutate(rawdata, contact_low = ifelse(contact_method == "PHONE SALE", 1, 0))

for(level in unique(rawdata$contact_method)){
  rawdata[paste("dummy", level, sep = "_")] <- ifelse(rawdata$contact_method == level, 1, 0)
}

summary(rawdata)

rawdata <- rename(rawdata, c("dummy_FAMILY SALE" = "dummy_FAMILY_SALE", "dummy_MALL SALE" = "dummy_MALL_SALE", "dummy_RETAIL SALE" = "dummy_RETAIL_SALE", "dummy_INTERNET SALE" = "dummy_INTERNET_SALE"))

rawdata <- mutate(rawdata, contact_mid = dummy_FAMILY_SALE + dummy_OTHER + dummy_RETAIL_SALE)
rawdata <- mutate (rawdata, contact_high = dummy_INTERNET_SALE + dummy_MALL_SALE)
rawdata <- rawdata[, -c(32:37)]

#code dummy variables for rate_plan
rawdata <- mutate(rawdata, rate_low = ifelse(rate_plan == "BASIC", 1, 0))

for(level in unique(rawdata$rate_plan)){
  rawdata[paste("dummy", level, sep = "_")] <- ifelse(rawdata$rate_plan == level, 1, 0)
}

summary(rawdata)

rawdata <- rename(rawdata, c("dummy_200 MINUTE" = "dummy_200_MINUTE", "dummy_300 MINUTE" = "dummy_300_MINUTE", "dummy_100 MINUTE" = "dummy_100_MINUTE"))
rawdata <- mutate(rawdata, rate_mid = dummy_200_MINUTE + dummy_300_MINUTE + dummy_UNLIMITED)
rawdata <- mutate(rawdata, rate_high = dummy_100_MINUTE + dummy_OTHER)
rawdata <- rawdata[, -c(35:40)]

#saved this as a csv
write.csv(rawdata,"/Users/gmm372/Dropbox/School/STAT 642/Final Project/cleandata_with_dummies.csv")
rawdata.dummy <- read.csv("C:/Users/gmm372/Dropbox/School/STAT 642/Final Project/cleandata_with_dummies.csv")

#retain only variables we will be using
nndata <- rawdata[, c(6:8, 10, 22:36 )]
#drop city, state, zip (region describes attributes)
#drop total paid (total invoice and total open remain)
#activated/renew dates dropped for customer lifetime stat
#age dropped, not telling


nndata <- scale(nndata)
nndata <- cbind(nndata, rawdata$churn)
nndata <- as.data.frame(nndata)
nndata <- rename(nndata, c("V20" = "churn"))


##logistic regression

rawdata.logistic <- rawdata[,-c(1,2,3,4,5,12,13,16,17)]
logfit <- glm(churn~., family=binomial(link=logit), data=rawdata.logistic)
logfitstep <- step(logfit)
#glm.fit: fitted probabilities numerically 0 or 1 occurred 
#glm.fit: algorithm did not converge 


#solution? use glmnet - does penalized regression

library(glmnet)

#glmnet only works on matrices - need to transform using model.matrix
rawdatamatrix <- model.matrix(churn ~ region + contract_fee + tot_open_amt + tot_paid_amt + num_invoices
                              + age_range + credit_approval + contact_method + rate_plan + renewal_num + activated_num
                              + custlife + loyalcust + debt, data=rawdata.logistic)

#drop intercept from model matrix, and define x and y variables
x <- rawdatamatrix[,-1]
x <- x[train,]
y <- rawdata.logistic[train,]

fit1 <-glmnet(x, y$churn, family="binomial")
print(fit1)
plot(fit1)
coef(fit1, s=0.01)

log1pred <- predict(fit1, x, s=0.005, type="class")

logtable <- table(log1pred, y$churn)
logtable
log1error=(logtable[1,2]+logtable[2,1])/223159
log1error
#95% accuracy

#now to try with cross validation:

log2 <- cv.glmnet(x, y$churn, family="binomial")
plot(log2)

log2$lambda.min
#0.00006
coef(log2, s = "lambda.min")

log2pred <- predict(log2, x, s="lambda.min", type="class")
log2tables <- table(log2pred, y$churn)
log2tables
log2error=(log2tables[1,2]+log2tables[2,1])/223159
log2error
#97% accuracy















# Predict on Acutal Testdata
#convert variables names to lowercase
names(rawtest)<-tolower(names(rawtest))

rawtest<-read.csv("/Users/ShireeXue/Desktop/Spring 2015/Data Mining/Final Project/testdata.csv")

# Convert months to months in number
mo2Num <- function(x) match(tolower(x), tolower(month.abb))

#Format date variable for customer lifetime calculation 
rawtest<-mutate(rawtest,renewal_month=mo2Num(renewal_month),activated_month=mo2Num(activated_month),dummday="01")
rawtest<- mutate(rawtest, renewal_date=paste(renewal_month,dummday,renewal_year,sep="/"),activated_date=paste(activated_month,dummday,activated_year,sep="/"))
rawtest$renewal_num <- strptime(rawtest$renewal_date, "%m/%d/%Y")
rawtest$activated_num <- strptime(rawtest$activated_date, "%m/%d/%Y")

#Convert year and month from number to factors
rawtest$activated_year<- as.factor(rawtest$activated_year)
rawtest$activated_month<- as.factor(rawtest$activated_month)
rawtest$renewal_year<- as.factor(rawtest$renewal_year)
rawtest$renewal_month<- as.factor(rawtest$renewal_month)
rawtest$renewal_num<- as.Date(rawtest$renewal_num)
rawtest$activated_num<- as.Date(rawtest$activated_num)

# Calculate the customer lifeteime, unit in weeks. 
rawtest<-mutate(rawtest,custlife=difftime(rawtest$renewal_num,rawtest$activated_num,units="weeks"))
rawtest$custlife<- as.numeric(rawtest$custlife)
#Create new variable "loyalcust", 1 is loyal customers who renewed ,0 is not.
rawtest$loyalcust<- ifelse(rawtest$custlife== 0, 0, 1)
rawtest$debt<- ifelse(rawtest$tot_open_amt== 0, 1, 0)

sapply(finaldata2,class)

finaldata2$activated_date<- as.factor(finaldata2$activated_date)
finaldata2$renewal_date<- as.factor(finaldata2$renewal_date)
finaldata2$debt<- as.factor(finaldata2$debt)
finaldata2$loyalcust<- as.factor(finaldata2$loyalcust)

#Remove redundant variables
finaldata2<-select(rawtest,-currentbalance,-customerid,-dummday,-zip,-city,-state,
               -renewal_month,-renewal_year,-activated_month,-activated_year,
               -renewal_num,-activated_num)

finalrf2 <- predict(myrf,newdata=finaldata2,type="response")
write.csv(finalrf2, "/Users/ShireeXue/Desktop/Spring 2015/Data Mining/Final Project/finaloutput_origin.csv")

