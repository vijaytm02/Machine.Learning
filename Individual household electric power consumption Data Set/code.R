home <- read.table("household_power_consumption.txt",sep = ";",header = TRUE ,na.strings=c("","NA"))
home <- home[,-c(1,2)]
home <- home[rowSums(is.na(home)) == 0,]
for(k in 5:20){
  start_time <- Sys.time()
  clusters <- kmeans(home[,], k,nstart=20)
  end_time <- Sys.time()
  tm <- start_time - end_time
  k_min[k-4,1] <- clusters$betweenss/clusters$totss
  k_min[k-4,2] <- tm
}
plot(k_min)
lines(k_min)
