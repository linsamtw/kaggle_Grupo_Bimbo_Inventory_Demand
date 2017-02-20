
	#install.packages("data.table")
	#install.packages("snow")
	#install.packages("glmnet")
	#install.packages("dplyr")
	#install.packages("moments")
	#install.packages("xgboost", repos=c("http://dmlc.ml/drat/", getOption("repos")), type="source")	

	library(data.table)
	library(xgboost)
	library(snow)
	library(glmnet)
	library(dplyr)
	library(moments)
	
	#windows
	#setwd("g:\\kaggle")
	#linux
	setwd("/media/linsam/74D2F8C6D2F88D9C/kaggle")

	main.train=fread("train.csv")
	main.test=fread("test.csv")

	#data 分割
	main.train.y = 
  	main.train %>% 
  	filter(.,Semana==9) %>%
  	data.table(.)
	
	#訓練 預測9
	main.train.x = 
	main.train %>% 
	filter(.,Semana %in% c(3:8)) %>%
	data.table(.)
	#實際 預測10
	main.train.x.2=	
	main.train %>% 
	filter(.,Semana %in% c(3:9)) %>%
	data.table(.)
	
	rm(main.train)
	gc()
#------------------------------------------------------------
	s=Sys.time()
	result1=pred.fun(main.train.x,main.train.x.2,5000000,main.test)
	e=Sys.time()
	e-s
	gc()

	#s=Sys.time()
	#result2=pred.fun(main.train.x11,main.train.x11.2,5000000,main.test.11)
	#e=Sys.time()
	#e-s
	#gc()
	
	#------------------------------------------------------------
	result3 = result1 %>% arrange(id)
	
	gc()
	write.csv(result3,"pred1.csv",row.names = FALSE)
























