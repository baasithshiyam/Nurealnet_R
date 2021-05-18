library(fpp)
library(MASS)
library(readxl)
library(neuralnet)
library(ggplot2)
library(reshape2)
library(gridExtra)
library(fpp2)
library(e1071)
library(openxlsx)
library(MLmetrics)
#Read the excel file final
################part 2 #########
############# BAASITH ##############

setwd("E:/2nd Year/2nd sem/DataMining/cw")
readDataUSD <- read_excel("ExchangeUSD.xlsx")
str(readDataUSD)
names(readDataUSD)[1] <- "year"
names(readDataUSD)[2] <- "day"
names(readDataUSD)[3] <- "USD"
###### data collection and normailzation
usd_val<-readDataUSD[,3]


normalize <- function(x) {
  return ((x - min(x)) / (max(x) - min(x)))
}


date_column <- factor(readDataUSD$year)
date_column <- as.numeric(date_column)
date_column


day_column <- factor(readDataUSD$day)
day_column <- as.numeric(day_column)
day_column

Exchange <- data.frame(date_column,day_column,readDataUSD$USD)
Exchange
str(Exchange)

Exchanged_scale <- as.data.frame(lapply(Exchange, normalize))
names(Exchanged_scale)[1] <- "date"
names(Exchanged_scale)[2] <- "day"
names(Exchanged_scale)[3] <- "USD"
str(Exchanged_scale)
summary(Exchange)
#Training a nn model
set.seed(123)
exchanged_scale_train <- Exchanged_scale[1:400,]
exchanged_scale_test <- Exchanged_scale[401:500,]
nn_exchanged_scale<- neuralnet(USD  ~ day + date  ,hidden=c(3,3) , data = exchanged_scale_train )
plot(nn_exchanged_scale)

model_output <- predict(nn_exchanged_scale, exchanged_scale_test[2:3])
model_output  #testing the model
str(model_output)

# extract the original (not normalized) training and testing desired Output from orginal data set
exchange_train_orginal<- usd_val[1:400,"USD"]  # the first 400 rows
exchange_test_original <- usd_val[401:500,"USD"] # the remainining rows

# and find its maximum & minimum value
original_min <- min(exchange_train_orginal)
original_max <- max(exchange_train_orginal)

# display its contents in the head
head(exchange_train_orginal)

#Create the reverse of normalized function - normalized
unnormalize <- function(x, min, max) { 
  return( (max - min)*x + min )
}

#Now we re-normalize the normalized neuralnet output nn_exchanged_scale 
renormalized_predict <- unnormalize(model_output, original_min, original_max)
renormalized_predict   # this is NN's output renormalized to original ranges

str(renormalized_predict)
########## Testing the accuracy  model 1##############
#Define RMSE function

RMSE(exp(renormalized_predict),exchange_test_original$USD)

#MSE
MSE(exp(renormalized_predict),exchange_test_original$USD)

#Define MAPE
MAPE(exp(renormalized_predict),exchange_test_original$USD)

# examine the correlation between predicted and actual values
cor(exp(renormalized_predict),exchange_test_original$USD)

#Plot for exchange_model
par(mfrow=c(1,1))
plot(exchange_test_original$USD, renormalized_predict ,col='green',main='Real vs predicted NN',pch=22,cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend='NN', pch=18,col='red', bty='n')

final_res <- cbind(exchange_test_original, renormalized_predict)
final_res

plot(exchange_test_original$USD , ylab = "Predicted vs Expected Model 1 ", type="l", col="blue" )
par(new=TRUE)
plot(renormalized_predict, ylab = " ", yaxt="n", type="l", col="green"  )
legend("bottomright",
       c("Predicted","Expected"),
       fill=c("green","blue")
)








# model 2
# Changing model Performance hidden=1,nodes=3 and tanh act.function

set.seed(213212)
nn_exchanged_scale.1<- neuralnet(USD  ~ day + date  , data = exchanged_scale_train ,hidden = c(3),act.fct = "logistic") 
plot(nn_exchanged_scale.1)
model_output.1 <- predict(nn_exchanged_scale.1, exchanged_scale_test[2:3])
model_output.1  #testing the model



#Now we renormalize the normalised neuralnet output nn_exchanged_scale 
renormalized_predict.1 <- unnormalize(model_output.1, original_min, original_max)
renormalized_predict.1 
head(renormalized_predict.1)# this is NN's output renormalized to original ranges


# Testing the accuracy model 2 

#Define RMSE function
RMSE(exp(renormalized_predict.1),exchange_test_original$USD)

#MSE
MSE(exp(renormalized_predict.1),exchange_test_original$USD)

#Define MAPE
MAPE(exp(renormalized_predict.1),exchange_test_original$USD)

# examine the correlation between predicted and actual values
cor(exp(renormalized_predict.1),exchange_test_original$USD)


#Plot for # 2 one
par(mfrow=c(1,1))
plot(exchange_test_original$USD, renormalized_predict.1 ,col='green',main='Real vs predicted NN',pch=22,cex=0.7)
points(exchange_test_original$USD,renormalized_predict,col='red',pch=18,cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend='NN', pch=18,col='red', bty='n')

final_res.1 <- cbind(exchange_test_original, renormalized_predict)
final_res.1


plot(exchange_test_original$USD , ylab = "Predicted vs Expected Model 2", type="l", col="blue" )
par(new=TRUE)
plot(renormalized_predict.1, ylab = " ", yaxt="n", type="l", col="green" )
legend("bottomright",
       c("Predicted","Expected"),
       fill=c("green","blue")
)





#Model 3 in 

set.seed(213213)
nn_exchanged_scale.2 <- neuralnet( USD  ~ day + date  , data = exchanged_scale_train , hidden = c(3,1) ,algorithm = "rprop-", act.fct='logistic') 
model_output.2 <- predict(nn_exchanged_scale.2,exchanged_scale_test[2:3])

plot(nn_exchanged_scale.2)

# this is NN's output renormalized to original ranges
renormalized_predict.2 <- unnormalize(model_output.2, original_min, original_max)
renormalized_predict.2 

# Testing the accuracy Model 3 
#Define RMSE function
RMSE(exp(renormalized_predict.2),exchange_test_original$USD)

#MSE
MSE(exp(renormalized_predict.2),exchange_test_original$USD)

#Define MAPE
MAPE(exp(renormalized_predict.2),exchange_test_original$USD)

# examine the correlation between predicted and actual values
cor(renormalized_predict.2,exchange_test_original$USD)

#Plot
par(mfrow=c(1,1))
plot(exchange_test_original$USD, renormalized_predict.2 ,col='red',main='Real vs predicted NN',pch=18,cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend='NN', pch=18,col='red', bty='n')
final_result2.2 <- cbind(exchange_test_original, renormalized_predict.2)
final_result2.2

plot(exchange_test_original$USD , ylab = "Predicted vs Expected model 3 ", type="l", col="blue" )
par(new=TRUE)
plot(renormalized_predict.2, ylab = " ", yaxt="n", type="l", col="green" )
legend("bottomright",
       c("Predicted","Expected"),
       fill=c("green","blue")
)





#Model 4 in 

set.seed(1221334)
nn_exchanged_scale.3 <- neuralnet( USD  ~ day + date  , data = exchanged_scale_train , hidden = c(3,2)) 
model_output.3 <- predict(nn_exchanged_scale.3,exchanged_scale_test[2:3])

plot(nn_exchanged_scale.3)

# this is NN's output renormalized to original ranges
renormalized_predict.3 <- unnormalize(model_output.3, original_min, original_max)
renormalized_predict.3 

# Testing the accuracy Model 4 
#Define RMSE function
RMSE(exp(renormalized_predict.3),exchange_test_original$USD)

#MSE
MSE(exp(renormalized_predict.3),exchange_test_original$USD)

#Define MAPE
MAPE(exp(renormalized_predict.3),exchange_test_original$USD)

# examine the correlation between predicted and actual values
cor(renormalized_predict.3,exchange_test_original$USD)

#Plot
par(mfrow=c(1,1))
plot(exchange_test_original$USD, renormalized_predict.3 ,col='red',main='Real vs predicted NN 3',pch=18,cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend='NN', pch=18,col='red', bty='n')
final_result2.3 <- cbind(exchange_test_original, renormalized_predict.3)
final_result2.3

plot(exchange_test_original$USD , ylab = "Predicted vs Expected model 4 ", type="l", col="blue" )
par(new=TRUE)
plot(renormalized_predict.3, ylab = " ", yaxt="n", type="l", col="green" )
legend("bottomright",
       c("Predicted","Expected"),
       fill=c("green","blue")
)






#forecasting using nnetar the future of USD/EUR 


class(readDataUSD)
library(tsbox)
keep <- c("year","USD") #delete the Wdy or day
df = readDataUSD[keep]
df #assign it to a new data frame
#install.packages("tsbox")
timeseriesDat<- ts_ts(ts_long(df))
class(timeseriesDat) 
str(timeseriesDat)
timeseriesDat = na.locf(timeseriesDat , fromLast = TRUE)
timeseriesDat = ts(timeseriesDat,
                    start = 2011,
                   frequency = 365)
modelnnetr = nnetar(timeseriesDat,PI=TRUE)
library(forecast)
forcastData = forecast(modelnnetr,h = 150)
plot(forcastData , ylab = "USD",xlab = "Year" , col= "red",main ='Forcast of USD/EUR') 



