data<-read.csv("bank.csv",stringsAsFactors=T,sep=";")
set.seed(12365)
a=nrow(data)
t=ceiling(a*0.7)
traindata=data[1:t,]
testdata=data[t:a,]
library(e1071)
model <- svm( y ~ .,  data = traindata, kernel="radial")
predictedY <- as.vector(predict(model, testdata))
sums <- 0
for(i in 1:nrow(testdata)){
  if (testdata[i,17]!=predictedY[i]){
    sums <- sums + 1
  }
}
sums <- sums/nrow(testdata)
print(paste("Accuracy rate is",(1-sums)*100))
