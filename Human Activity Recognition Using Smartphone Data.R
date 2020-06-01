###########################################################

# Load libraries
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(RColorBrewer)) install.packages("RColorBrewer", repos = "http://cran.us.r-project.org")
if(!require(matrixStats)) install.packages("matrixStats", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(Rborist)) install.packages("Rborist", repos = "http://cran.us.r-project.org")
if(!require(rpart)) install.packages("rpart", repos = "http://cran.us.r-project.org")
if(!require(klaR)) install.packages("klaR", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require(rpart.plot)) install.packages("rpart.plot", repos = "http://cran.us.r-project.org")
options(digits = 3)

# Data has been downloaded from Kaggle and stored in project folder
# https://www.kaggle.com/uciml/human-activity-recognition-with-smartphones
# Importing training and test dataset from the zip file

zipfile <- "./smartlab_dataset/human-activity-recognition-with-smartphones.zip"
unzip(zipfile, list = TRUE)$Name # check the file names

train_set <- read.csv(unzip(zipfile, files = "train.csv"))
test_set <- read.csv(unzip(zipfile, files = "test.csv"))
x <- train_set[,1:561] # training features
y <- train_set$Activity # Activity vector

# Activities & subjects
activities <- levels(y)
unique(train_set$Activity)
unique(test_set$Activity)
length(unique(train_set$subject))
length(unique(test_set$subject))


# Data distribution visualization
train_set %>% rbind(test_set) %>% group_by(subject, Activity) %>% 
  summarize(count= n()) %>%
  ggplot(aes(subject, count,fill=Activity)) +
  geom_bar(position = "stack", stat = "identity") +
  scale_fill_brewer(palette="Paired") +
  theme(legend.position="bottom") 

# data range
range(train_set[,1:561])

# visualizing features 

train_set %>% 
  ggplot(aes(tGravityAcc.energy...Y, angle.Z.gravityMean.,color=Activity)) +
  geom_point() +
  scale_color_brewer(palette="Paired") +
  theme(legend.position="bottom") 

train_set %>% 
  ggplot(aes(tGravityAcc.sma.., tGravityAcc.energy...Z, 
             color=Activity)) +
  geom_point() +
  scale_color_brewer(palette="Paired") +
  theme(legend.position="bottom") 

train_set %>% 
  ggplot(aes(tBodyAcc.correlation...X.Y, fBodyGyro.bandsEnergy...9.16, 
             color=Activity)) +
  geom_point() +
  scale_color_brewer(palette="Paired") +
  theme(legend.position="bottom") 

# distance for 200 x 200 sampled data
set.seed(123, sample.kind = "Rounding")
n <- 200
rindex <- sample(nrow(x), n)
cindex <- sample(ncol(x), n)
d <- dist(x[rindex,cindex]) # feature distance
image(as.matrix(d)[order(y[rindex]), order(y[rindex])], 
      col = rev(RColorBrewer::brewer.pal(9, "RdBu"))) # display distance in order of activity

# Pre-processing
# near-zero variance predictors
nzv <- nearZeroVar(x, saveMetrics= TRUE)
sum(nzv$zeroVar) # no of Zero Variance 
sum(nzv$nzv) # no of nzv

range(nzv$percentUnique)

# correlated predictors
xCor <- cor(as.matrix(x)) #create correlation matrix
summary(xCor[upper.tri(xCor)]) # Show the summary
highlyCorX <- findCorrelation(xCor, cutoff = .75)
x_lowCor <- x[,-highlyCorX] # filtern out features with cor > .75
dim(x_lowCor)

# recreate training set using filtered predictors
ftrain_set <- data.frame(y, as.data.frame(x_lowCor))
save(ftrain_set, file = "ftrain_set.RData")

# adjust the test set by filtering the feature
tx <- as.matrix(test_set[,1:561])
ty <- test_set$Activity
tx_lowCor <- tx[,-highlyCorX] # filter out features with cor > .75
ftest_set <- data.frame(y=ty, as.data.frame(tx_lowCor))
save(ftest_set, file = "ftest_set.RData")

############################# Algorithm Training and Testing #################
########## knn #########
# train using filtered training data
# create data subset for test run

set.seed(1, sample.kind = "Rounding")
n <- 1000
b <- 5
index <- sample(nrow(x_lowCor), n)
control <- trainControl(method="cv", number = b, p = .9)
train_knn <- train(x_lowCor[index, ], y[index], method = "knn", 
                   trControl = control,
                   tuneGrid = data.frame(k = seq(1, 30, 2)))

# Visualize optimum k
ggplot(train_knn, highlight = TRUE)
train_knn$bestTune
save(train_knn, file = "train_knn.RData")

# apply to entire dataset
fit_knn <- knn3(x_lowCor, y,  k = train_knn$bestTune$k) 

# apply for prediction (using filtered test predictors)
y_hat_knn <- predict(fit_knn, tx_lowCor, type ="class")

# generate confusion matrix
knn_CM <- confusionMatrix(y_hat_knn, ftest_set$y)
knn_CM$overall["Accuracy"]
knn_CM

# create accuracy table
accuracy_results <- data.frame(Method = "KNN", Accuracy = knn_CM$overall["Accuracy"])

####### QDA ########
# QDA plot
partimat(y ~ angle.Z.gravityMean. + tGravityAcc.energy...Y,
         data = ftrain_set, method= "qda")

# train using entire training set
tcontrol <- trainControl(method="cv", number = 5, p = .9)
train_qda <- train(x_lowCor, y,
                   method = "qda", 
                   trControl = tcontrol)

train_qda

y_hat_qda <- predict(train_qda, tx_lowCor, type ="raw")
qda_CM <- confusionMatrix(y_hat_qda, ftest_set$y)
qda_CM$overall["Accuracy"]

# update accuracy table
accuracy_results <- accuracy_results %>%
  rbind(data.frame(Method = "QDA", Accuracy = qda_CM$overall["Accuracy"]))

####### Rpart Decision Tree  ########
# sample the training data to prevent long runtime
set.seed(555, sample.kind = "Rounding")
n <- 2000
index <- sample(nrow(x_lowCor), n)
# cp optimization
tcontrol <- trainControl(method="cv", number = 10)
tngrid <- expand.grid(cp = seq(0, 0.02, 0.001))
train_rpart = train(x_lowCor[index,], y[index], 
                    method = "rpart",
                    trControl = tcontrol,
                    tuneGrid = tngrid)

ggplot(train_rpart, highlight = TRUE)
train_rpart
save(train_rpart, file = "train_rpart.RData")

# maxdepth optimization
tngrid2 <- expand.grid(maxdepth = seq(5, 30, 5))
train_rpart2 = train(x_lowCor[index,], y[index], 
                     method = "rpart2",
                     trControl = tcontrol,
                     tuneGrid = tngrid2)

ggplot(train_rpart2, highlight = TRUE)
train_rpart2
save(train_rpart2, file = "train_rpart2.RData")

# generate final model using tuned parameters
fit_rpart <- rpart(y ~ ., data = ftrain_set, 
                   control = rpart.control(
                     cp = train_rpart$bestTune$cp,
                     maxdepth = train_rpart2$bestTune$maxdepth
                   ))
# plot the tree
prp(fit_rpart, type = 0, cex = 0.6)
save(fit_rpart, file = "fit_rpart.RData")

# apply for prediction (using filtered test predictors)
y_hat_rpart <- predict(fit_rpart, ftest_set, type ="class")
rpart_CM <- confusionMatrix(y_hat_rpart, ftest_set$y)
rpart_CM$overall["Accuracy"]
rpart_CM

# update accuracy table
accuracy_results <- accuracy_results %>%
  rbind(data.frame(Method = "RPART", Accuracy = rpart_CM$overall["Accuracy"]))

######### Random Forest #########
# optimization - mtry
set.seed(777, sample.kind = "Rounding")
n <- 2000
index <- sample(nrow(x_lowCor), n)
tngrd <- data.frame(mtry = c(12, 40, 50, 60))
tcontrol <- trainControl(method="cv", number = 5)
train_rf <-  train(x_lowCor[index,], y[index], 
                   method = "rf", 
                   ntree = 200,
                   trControl = tcontrol,
                   tuneGrid = tngrd,
                   nSamp = 1000)

ggplot(train_rf, highlight=TRUE)
train_rf$bestTune
save(train_rf, file = "train_rf.RData")

# optimization - nodeSize using the same sample of observation
tngrd <- data.frame(predFixed= 2, minNode = c(1, 5, 10, 25))
control <- trainControl(method="cv", number = 5)
train_rb <-  train(x_lowCor[index,], y[index], 
                   method = "Rborist", 
                   trControl = control,
                   tuneGrid = tngrd)

ggplot(train_rb, highlight = TRUE)
train_rb$bestTune
save(train_rb, file = "train_rb.RData")

# generate final model using tuned parameters
fit_rf <- randomForest(x_lowCor, y, 
                       mtry = train_rf$bestTune$mtry,
                       nodesize = train_rb$bestTune$minNode,
                       importance = TRUE)

# check for enough trees
plot(fit_rf)
as.data.frame(fit_rf$err.rate) %>% 
  mutate(trees = seq.int(nrow(.))) %>% 
  gather(key = "key", value = "error", -trees) %>%
  ggplot(aes(x=trees, y = error)) +
  geom_line(aes(color = key), size = 1) +
  theme(legend.position="top") +
  scale_color_brewer(palette="Paired")

save(fit_rf, file = "fit_rf.RData")

# apply for prediction (using filtered test predictors)
y_hat_rf <- predict(fit_rf, tx_lowCor, type ="class")
rf_CM <- confusionMatrix(y_hat_rf, ftest_set$y)
rf_CM$overall["Accuracy"]

# variable importance
rf_imp <- varImp(fit_rf)
varImpPlot(fit_rf, n.var = 10, cex= 0.75)

# update accuracy table
accuracy_results <- accuracy_results %>%
  rbind(data.frame(Method = "Random Forest", Accuracy = rf_CM$overall["Accuracy"]))
knitr::kable(accuracy_results)
save(accuracy_results, file="accuracy_results.RData")

# Visualize accuracy
accuracy_results %>%
  ggplot(aes(x = Method, fill = Method)) +
  geom_bar(aes(y = Accuracy), stat = "identity", ) +
  ylim(0, 1) + 
  geom_text(aes(y = Accuracy, label=sprintf("%0.2f", round(Accuracy, digits = 2)),
                vjust = -0.5, size = 2)) +
  scale_fill_brewer(palette="Dark2")

# Sensitivity table
sensitivity_table <- data.frame(Method = "KNN", t(knn_CM[["byClass"]][,"Sensitivity"])) %>%
  rbind(data.frame(Method = "QDA", t(qda_CM[["byClass"]][,"Sensitivity"]))) %>%
  rbind(data.frame(Method = "RPART", t(rpart_CM[["byClass"]][,"Sensitivity"]))) %>%
  rbind(data.frame(Method = "Random Forest", t(rf_CM[["byClass"]][,"Sensitivity"])))
sensitivity_table <- sensitivity_table %>% `colnames<-`(c("Method", "Laying", "Sitting", " Standing", 
                                                          "Walking", "Walking Downstairs", "Walking Upstairs"))
sensitivity_table
save(sensitivity_table, file = "sensitivity_table.RData")

# Visualize sensitivity
sensitivity_table %>% 
  gather (key = "Class", value = "Sensitivity", -Method) %>%
  ggplot() +
  geom_bar(aes(x = Class, y = Sensitivity, fill = Method), 
           stat = "identity", position = "dodge") +
  geom_text(aes(x = Class, y = Sensitivity, fill = Method,
                label=sprintf("%0.2f", round(Sensitivity, digits = 2))),
            position = position_dodge(width = 1), 
            vjust = -0.5, size = 1.75) +
  ylim(0, 1) + 
  scale_fill_brewer(palette="Dark2") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# output CM tables for QDA & RF
qda_cm_tab <- qda_CM[["table"]] %>%
  `colnames<-`(c("Laying", "Sitting", "Standing", 
                 "Walking", "W_Downstairs", "W_Upstairs")) %>%
  `rownames<-`(c("Laying", "Sitting", "Standing", 
                 "Walking", "W_Downstairs", "W_Upstairs"))
qda_cm_tab
save(qda_cm_tab, file = "qda_cm_tab.RData")
rf_cm_tab <- rf_CM[["table"]] %>%
  `colnames<-`(c("Laying", "Sitting", "Standing", 
                 "Walking", "W_Downstairs", "W_Upstairs")) %>%
  `rownames<-`(c("Laying", "Sitting", "Standing", 
                 "Walking", "W_Downstairs", "W_Upstairs"))
save(rf_cm_tab, file = "rf_cm_tab.RData")

# Specificity table
specificity_table <- data.frame(Method = "KNN", t(knn_CM[["byClass"]][,"Specificity"])) %>%
  rbind(data.frame(Method = "QDA", t(qda_CM[["byClass"]][,"Specificity"]))) %>%
  rbind(data.frame(Method = "RPART", t(rpart_CM[["byClass"]][,"Specificity"]))) %>%
  rbind(data.frame(Method = "Random Forest", t(rf_CM[["byClass"]][,"Specificity"])))  %>% 
  `colnames<-`(c("Method", "Laying", "Sitting", " Standing", 
                 "Walking", "Walking Downstairs", "Walking Upstairs"))
knitr::kable(specificity_table)
save(specificity_table, file = "specificity_table.RData")

# Visualize specificity
specificity_table %>% 
  gather (key = "Class", value = "Specificity", -Method) %>%
  ggplot() +
  geom_bar(aes(x = Class, y = Specificity, fill = Method), 
           stat = "identity", position = "dodge") +
  geom_text(aes(x = Class, y = Specificity, fill = Method,
                label=sprintf("%0.2f", round(Specificity, digits = 2))),
            position = position_dodge(width = 1), 
            vjust = -0.5, size = 1.75) +
  scale_fill_brewer(palette="Dark2") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Balanced Accuracy
bal_accuracy_table <- data.frame(Method = "KNN", t(knn_CM[["byClass"]][,"Balanced Accuracy"])) %>%
  rbind(data.frame(Method = "QDA", t(qda_CM[["byClass"]][,"Balanced Accuracy"]))) %>%
  rbind(data.frame(Method = "RPART", t(rpart_CM[["byClass"]][,"Balanced Accuracy"]))) %>%
  rbind(data.frame(Method = "Random Forest", t(rf_CM[["byClass"]][,"Balanced Accuracy"])))  %>% 
  `colnames<-`(c("Method", "Laying", "Sitting", " Standing", 
                 "Walking", "Walking Downstairs", "Walking Upstairs"))
knitr::kable(bal_accuracy_table)
save(bal_accuracy_table, file = "bal_accuracy_table.RData")

# Visualize Balanced Accuracy 
bal_accuracy_table %>% 
  gather (key = "Class", value = "Balanced_Accuracy", -Method) %>%
  ggplot() +
  geom_bar(aes(x = Class, y = Balanced_Accuracy, fill = Method), 
           stat = "identity", position = "dodge") +
  geom_text(aes(x = Class, y = Balanced_Accuracy, fill = Method,
                label=sprintf("%0.2f", round(Balanced_Accuracy, digits = 2))),
            position = position_dodge(width = 1), 
            vjust = -0.5, size = 1.75) +
  ylim(0, 1) + 
  scale_fill_brewer(palette="Dark2") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

###########################################################