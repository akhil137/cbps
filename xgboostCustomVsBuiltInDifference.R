#Code below to demo difference between custom objective and builtin logistic
library(xgboost)
#generate synthetic data
set.seed(1979)
#num units/records
n=100
#num covariates
ncov=4
z=matrix(replicate(n,rnorm(ncov)),nrow=n)
alpha=c(-1,0.5,-0.25,-0.1)
za=z%*%alpha
p=exp(za)/(1+exp(za))
t=rbinom(n,1,p)
dtrain<-xgb.DMatrix(data=z,label=t)


#Custom objective and error 
logregobj <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  preds <- 1/(1 + exp(-preds))
  grad <- preds - labels
  hess <- preds * (1 - preds)
  return(list(grad = grad, hess = hess))
}

evalerror <- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  err <- as.numeric(sum(labels != (preds > 0)))/length(labels)
  return(list(metric = "error", value = err))
}

#Create model based on custom objective above
param<-list(objective=logregobj, eval_metric=evalerror)
bstCustom <- xgb.train(param,dtrain, nrounds=10)

#Create model based on built-in objective
bstCheck <- xgb.train(list(objective="binary:logitraw"),dtrain, nrounds=10)
#Error between two methods larger than expected
sum(abs(predict(bstCustom,z)-predict(bstCheck,z)))