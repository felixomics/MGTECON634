# there is a change
# 2 changes
#####################################
# Reading processed datasets into R #
#####################################

#install.packages("AER")
#install.packages("MatchIt")
#install.packages("twang")
library(MASS)
library(AER)
library(devtools)
library(randomForest) 
library(rpart) # decision tree
library(rpart.plot) # enhanced tree plots
library(rattle) # fancy tree plot
library(ROCR)
library(Hmisc)
library(corrplot)
library(texreg)
library(glmnet)
library(reshape2)
library(knitr)
library(xtable)
library(lars)
library(ggplot2)
library(matrixStats)
library(plyr)
library(doMC)
library(stargazer)
registerDoMC(cores=4) # for a simple parallel computation
load("/Users/apple/360yunpan/MGTECON 634/rdata/mobil.rdata")
# Add these noise covariates to the social data
working <- mobilization

# Pick a selection of variables
variables.names <- c("persons","competiv","vote00","vote98","newreg","age","female","state",
                     "comp_mi","comp_ia","lc","hc",
                     "vote02", "pseudo_c", "treat_pseudo")
# now no need to worry about NAs
working.nomissing <- na.omit(working[variables.names])

# get a small sample, since data is too big
#set.seed(333)
#working <- working.nomissing[sample(nrow(working.nomissing), 20000), ]
working <- working.nomissing

#############################
###### drop observations ####
#############################
working<-working[-which(working$treat_pseudo == 0 & working$age >=85),]
working<-working[-which(working$treat_pseudo == 1 & working$age < 35),]

#########################################################
# We generate noise covariates and add them in the data #
#########################################################
set.seed(123)
noise.covars <- matrix(data = runif(nrow(working) * 13), 
                       nrow = nrow(working), ncol = 13)
noise.covars <- data.frame(noise.covars)
names(noise.covars) <- c("noise1", "noise2", "noise3", "noise4", "noise5", "noise6",
                         "noise7", "noise8", "noise9", "noise10", "noise11", "noise12","noise13")

# Add these noise covariates to the mobilization data
working <- cbind(working, noise.covars)
#working <- working[sample(nrow(working), 100000), ]
working.t<-working[which(working$treat_pseudo ==1),]
working.c<-working[which(working$treat_pseudo ==0),]
set.seed(345)
working<-rbind(working.t[sample(nrow(working.t),50000),],working.c[sample(nrow(working.c),50000),])

# Pick a selection of covariates
covariate.names <- c("persons","competiv","vote00","vote98","newreg","age","female","state",
                     "comp_mi","comp_ia","lc","hc",
                     "noise1", "noise2", "noise3", "noise4", "noise5", "noise6",
                     "noise7", "noise8", "noise9", "noise10", "noise11", "noise12","noise13")

# The dependent (outcome) variable is whether the person voted, 
# so let's rename "outcome_voted" to Y
names(working)[names(working)=="vote02"] <- "Y"

# Extract the dependent variable
Y <- working[["Y"]]

# pseudo_c is whether they are treated
names(working)[names(working)=="pseudo_c"] <- "W"

# treat_pseudo is whether they are assigned to treatment group
names(working)[names(working)=="treat_pseudo"] <- "assignment"

# Extract treatment variable & covariates
W <- working[["W"]]
assignment <- working[["assignment"]]
covariates <- working[covariate.names]

# some algorithms require our covariates be scaled
# scale, with default settings, will calculate the mean and standard deviation of the entire vector, 
# then "scale" each element by those values by subtracting the mean and dividing by the sd
covariates.scaled <- scale(covariates)
processed.scaled <- data.frame(Y, W, covariates.scaled)
processed.unscaled <- data.frame(Y, W, covariates)
set.seed(44)
smplmain <- sample(nrow(processed.scaled), round(9*nrow(processed.scaled)/10), replace=FALSE)

processed.scaled.train <- processed.scaled[smplmain,]
processed.scaled.test <- processed.scaled[-smplmain,]
############################
#  simple ols y on W and X #
############################
# Creating Formulas
covariate.chosen <- c("persons","competiv","vote00","vote98","newreg","age","female","state",
                      "comp_mi","comp_ia",
                      "noise1", "noise2", "noise3", "noise4", "noise5", "noise6",
                      "noise7", "noise8", "noise9", "noise10", "noise11", "noise12","noise13")
sumx1 = paste(covariate.chosen, collapse = " + ")  # "X1 + X2 + X3 + ..." for substitution later
# Y ~ W + X1 + X2 + ... 
before <- paste("W",sumx1, sep=" + ")
all <- paste("Y", before, sep=" ~ ")
ols1 <- as.formula(all)
# create formula
ols1
# regression
ols_results1 <- lm(ols1, processed.unscaled)
# summary
summary(ols_results1)
# CI
confint(ols_results1, 'W', level=0.95)


#############################
# propensity score weighting #
##############################
# logit of W on x's
# Creating Formulas
# W ~ X1 + X2 + ... 
all <- paste("W", sumx1, sep=" ~ ")
ps <- as.formula(all)
ps

current.working <- processed.unscaled
# logit reg
glm1 <- glm(ps, family=binomial, current.working)
# estimated propensity scores
summary(glm1)
# fitted value of logit reg
current.working$weight <- glm1$fitted
# plot the distribution of ps
hist(current.working$weight, main = "Propensity Score Estimation")
# weights
current.working$weight.ATE <- ifelse(current.working$W == 1, 1/current.working$weight,
                             1/(1-current.working$weight))
# weighted regression
psw1 <- lm(Y ~ W, data=current.working, weights=(weight.ATE))
summary(psw1)
confint(psw1, 'W', level=0.95)

######################################
# lasso to estimate propensity score #
######################################
current.working <- processed.scaled
# generate x and y
x <- as.matrix(current.working[covariate.chosen])
y <- as.matrix(current.working["W"])
# use lasso to select important variables for w~x
lasso.logit <- cv.glmnet(x, y,  alpha=1, family='binomial')
plot(lasso.logit)
grid()
# use the best lambda
lasso.logit$lambda.min
lasso.logit$lambda.1se
# grab the predicted values
coef1 <- predict(lasso.logit, type = "nonzero") # Method 2

# index the column names of the matrix in order to index the selected variables
colnames <- colnames(x)
selected.vars1 <- colnames[unlist(coef1)]
# output the selected variables
selected.vars1

# propensity score weighting
# logit, W on x's
logistic <- paste("W", paste(selected.vars1,collapse=" + "), sep = " ~ ") 
logistic <- as.formula(logistic)
# logit regression by using lasso selected var
glm2 <- glm(logistic, family=binomial, data=current.working)
# estimated propensity scores
summary(glm2)
# fitted values
current.working$weight2 <- glm2$fitted
# weights
current.working$weight2.ATE <- ifelse(current.working$W == 1, 1/current.working$weight2,
                                      1/(1-current.working$weight2))
# weighted regression
psw2 <- lm(Y ~ W, data=current.working, weights=(weight2.ATE))
summary(psw2)
# CI
confint(psw2, 'W', level=0.95)

##############################
### single lasso ATE #########
##############################
current.working <- processed.scaled
#x<-model.matrix(ols1, processed.scaled.train)[,-1]
x <- as.matrix(current.working[c( "W", covariate.chosen)])
#generate y vector
y <- as.matrix(current.working["Y"])

p.fac = rep(1, 26) # put 0 penalty on W
p.fac[c(1)] = 0
#lasso.logit <- cv.glmnet(x, y, penalty.factor = p.fac, alpha=1, family='binomial')
# directly predict the ATE
lasso.logit1 <- cv.glmnet(x, y, penalty.factor = p.fac, alpha=1)

lasso.logit
plot(lasso.logit)
grid()
# best lambda
lasso.logit$lambda.min
lasso.logit$lambda.1se
coef2 <- predict(lasso.logit, type = "nonzero") # Method 2
coef(lasso.logit, s="lambda.1se")

colnames <- colnames(x)
selected.vars2 <- colnames[unlist(coef2)]
# output selected results.
selected.vars2

#################################
## Belloni Chernozhukov Hansen ##
#################################
# BCH
covariate.lassochosen <- union(selected.vars1, selected.vars2)
covariate.lassochosen <-covariate.lassochosen[covariate.lassochosen!="W"]
sumx2 = paste(covariate.lassochosen, collapse = " + ")  # "X1 + X2 + X3 + ..." for substitution later
# Y ~ W + X1 + X2 + ... 
before <- paste("W",sumx2, sep=" + ")
all <- paste("Y", before, sep=" ~ ")
ols2 <- as.formula(all)
ols2
# regression
ols_results2 <- lm(ols2, processed.unscaled)
summary(ols_results2)
confint(ols_results2, 'W', level=0.95)



#########################################
## how ATE changes with regularization ##
#########################################

current.working <- processed.scaled
x <- as.matrix(current.working[c( "W", covariate.chosen)])
y <- as.matrix(current.working["Y"])

p.fac = rep(1, 26) # put 0 penalty on W
p.fac[c(1)] = 0

fit <- glmnet(x, y, penalty.factor = p.fac, alpha=1, family='binomial')

plot(fit, label=TRUE)


###############################################
## set penalty 0 for lasso chosen parameters ##
###############################################
current.working <- processed.scaled
x <- as.matrix(current.working[c( "W", covariate.chosen)])
y <- as.matrix(current.working["Y"])

p.fac = rep(1, 26) # put 0 penalty on W
p.fac[c(1, 6, 8, 9, 10)] = 0

fit <- glmnet(x, y, penalty.factor = p.fac, alpha=1, family='binomial')

plot(fit, label=TRUE)


library(devtools) 
install_github("swager/balanceHD")
library(balanceHD)
###########################
### Residual Balance ATE###
###########################
current.working <- processed.scaled[sample(length(processed.scaled$Y),4000),]
X <- as.matrix(current.working[covariate.chosen])
W <- as.matrix(current.working["W"])
Y <- as.matrix(current.working["Y"])
# ordinary residualBalance.ate
tau.hat = residualBalance.ate(X, Y, W,estimate.se = TRUE)
# with lasso PSW
tau.hat1 = residualBalance.ate(X, Y, W,fit.method = "elnet", estimate.se = TRUE)
# case 1
print(paste("point estimate:", round(tau.hat[1], 4)))
print(paste0("95% CI for tau: (", round(tau.hat[1] - 1.96 * tau.hat[2], 3), ", ", round(tau.hat[1] + 1.96 * tau.hat[2], 3), ")"))
# case 2
print(paste("point estimate:", round(tau.hat1[1], 4)))
print(paste0("95% CI for tau: (", round(tau.hat1[1] - 1.96 * tau.hat1[2], 3), ", ", round(tau.hat1[1] + 1.96 * tau.hat1[2], 3), ")"))


####################################
## CART estimate propensity score ##
####################################

set.seed(44)
# 10-fold
smplmain <- sample(nrow(processed.scaled), round(9*nrow(processed.scaled)/10), replace=FALSE)

processed.scaled.train <- processed.scaled[smplmain,]
processed.scaled.test <- processed.scaled[-smplmain,]

y.train <- as.matrix(processed.scaled.train$Y, ncol=1)
y.test <- as.matrix(processed.scaled.test$Y, ncol=1)


# Classification Tree with rpart
# grow tree 
set.seed(444)

linear.singletree <- rpart(formula = ols1, data=processed.scaled.train, 
                           method = "anova", y=TRUE,
                           control=rpart.control(cp=0.00001, minsplit=30))

printcp(linear.singletree) # display the results 
plotcp(linear.singletree) # visualize cross-validation results 

# prune the tree
op.index <- which.min(linear.singletree$cptable[, "xerror"])
cp.vals <- linear.singletree$cptable[, "CP"]
treepruned.linearsingle <- prune(linear.singletree, cp = cp.vals[op.index])

# apply model to the test set to get predictions
singletree.pred.class <- predict(treepruned.linearsingle, newdata=processed.scaled.test)

# plot tree 
#plot(treepruned.linearsingle, uniform=TRUE, 
#     main="Classification Tree Example")
#text(treepruned.linearsingle, use.n=F, all=TRUE, cex=.8)

plot(treepruned.linearsingle, uniform = T, compress = T, main="CART Propensity Score",margin = 0.001, branch = 0.5)
text(treepruned.linearsingle, use.n = T, digits = 3, cex = 0.9)

# weighted by CART propensity score

ps.cart <- lm(Y ~ W, data=processed.scaled.test, weights=(singletree.pred.class))
summary(ps.cart)
confint(ps.cart, 'W', level=0.95)


# Reference: Jin Chen, Kaiji Gong, Rui Xu, Zeyu Jin


