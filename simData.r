#sim propensity scores with logistic model form Kang/Schaefer 2008
set.seed(1979)
#num units/records
n=100
#num covariates
ncov=4
#iid of standard normals
z=matrix(replicate(n,rnorm(ncov)),nrow=n)
#vector of coeffs for prop score generation (Kang/Schaefer)
alpha=c(-1,0.5,-0.25,-0.1)
#mult and expit for 'true' prop scores
za=z%*%alpha
p=exp(za)/(1+exp(za))
#treatment is binomial with probability p
t=rbinom(n,1,p)
#make into a dat
dat<-data.frame(cbind(t,z))
#now logitstically regress
lgfit<-glm(dat$t~dat$V2+dat$V3+dat$V4+dat$V5,data=dat,family=binomial())
#predict prop scores frome it
pexpit<-predict(lgfit,type="response")

#xgboost
library(xgboost)
bst<-xgboost(data=z,label=t,nrounds=10,objective="binary:logistic")

#the predictions are 'scores' in xgboost lingo, i.e. estimated prob of being treated
pxgb=predict(bst,z)
#estimated treatment indicator
txgb=pxgb>0.5
#xgboost only misses 6
numIncorrect=sum(abs(t-txgb))

#Note that glm model get's more incorrect than xgb, but 
#its the same num incorrect as you would get using real 
#probabilities 
numIncorrect_real=sum(abs(t-(p>0.5)))
numIncorrect_glm=sum(abs(t-(pexpit>0.5)))

#however, the actual prop scores are closer between glm and real
sum(abs(pexpit-p))
sum(abs(pxgb-p))


#CBPS
library(CBPS)
cbfit<-CBPS(dat$t~dat$V2+dat$V3+dat$V4+dat$V5,data=dat)

#plot p-scores from each method
#make dataframe
pscores<-data.frame(meth=factor(rep(c("expit","xgb","cbps"), each=300)),p=c(pexpit,phat,cbfit$fitted.values))


#dithered hist
library(ggplot2)
ggplot(pscores,aes(p,fill=meth)) + geom_histogram(binwidth=0.05,alpha=0.5,position="dodge")

#boxplot (with pscores on x-axis)
ggplot(pscores, aes(x=meth, y=p, fill=meth)) + geom_boxplot() + guides(fill=FALSE) + coord_flip()	

#some nicer plots
library(GGally)
ggpairs(pscores)


#testing xgboost custom objective
#Note there are 2 options for prediction output: before (\hat{y}) or after (p) logistic tranformation 

#In the objective function though 
# user define objective function, given prediction, return gradient and second order gradient
# this is loglikelihood loss; note here preds is assumed to be data before expit
logregobj <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  preds <- 1/(1 + exp(-preds))
  grad <- preds - labels
  hess <- preds * (1 - preds)
  return(list(grad = grad, hess = hess))
}

# user defined evaluation function, return a pair metric_name, result
# NOTE: when you do customized loss function, the default prediction value is margin
# this may make buildin evalution metric not function properly
# for example, we are doing logistic loss, the prediction is score before logistic transformation
# the buildin evaluation error assumes input is after logistic transformation
# Take this in mind when you use the customization, and maybe you need write customized evaluation function
evalerror <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  err <- as.numeric(sum(labels != (preds > 0)))/length(labels)
  return(list(metric = "error", value = err))
}

#param <- list(max_depth=2, eta=1, nthread = 2, silent=1, 
#              objective=logregobj, eval_metric=evalerror)
print ('start training with user customized objective')
# training with customized objective, we can also do step by step training
# simply look at xgboost.py's implementation of train
bstRaw<-xgboost(data=z,label=t,nrounds=10,objective="binary:logitraw")
yhatRaw<-predict(bstRaw,z)
phatRaw<-1/(1 + exp(-yhatRaw))
#note very similar to what we got previously from obje=binary:logistic
sum(abs(pxgb-phatRaw))
#Also note that we can use logistic distributions quantile and cdf to go back/fort
#plogis(yhatRaw)=phatRaw (the CDF, inverse logit, expit, logistic transformation, or sigmoid)
#qlogis(phatRaw)=yhatRaw (logit); note it's less accurate

#now train with custom objective
dtrain<-xgb.DMatrix(data=z,label=t)
param<-list(objective=logregobj, eval_metric=evalerror)
bstCustom <- xgb.train(param,dtrain, nrounds=10)

#we can check if the call to xgb.train or xgboost matters
bstCheck <- xgb.train(list(objective="binary:logitraw"),dtrain, nrounds=10)
#no diff between the two calls 
sum(abs(predict(bstCheck,z)-predict(bstRaw,z))) 

#however there is a difference in the propensity scores between custom and built-in objective
#no difference between builtin-raw called with xgb.train and builtin-logistic called with xgboost
sum(abs(plogis(predict(bstCheck,z))-predict(bst,z))) #1e-6 error over all samples
#much bigger error with custom
sum(abs(plogis(predict(bstCustom,z))-predict(bst,z))) #about 3.5 over all samples, which is huge







