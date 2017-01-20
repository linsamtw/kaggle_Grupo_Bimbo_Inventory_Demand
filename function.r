
kaggle_fun=function(y,pred_y){

	n=length(y)
	pred_y2=round(pred_y)
	v1 = log(y+1)-log(pred_y2+1)
	v2 = v1^2
	v3 = sum(v2)
	v4 = sqrt(v3/n)

	return(v4)
}

work.train.test.map.fun2=function(n,amount){
	#n=nrow(train.x)
	num=c(1:n)	
	if( n*0.7>=amount){
		n1 = sample(num,amount)
	}else if( n*0.7<amount){
		n1 = sample(num,n*0.7)		
	}	
	n2=num[-n1]
	return(list(n1,n2))
}

xgb.fun=function(train,test3,nro,eta=0.1,md=10,cb=0.5){

	clf <- xgb.train(params=list(  objective="reg:linear", 
                               booster = "gbtree",
                               eta=eta, 
                               max_depth=md, 
                               colsample_bytree=cb) ,
                	data = xgb.DMatrix(
				data=data.matrix(
				train[,c(8:ncol(train)),with=FALSE]),
                	label=data.matrix(train[,.(log.due)])
				,missing=NA), 
				nrounds =nro,
                 	verbose = 1,
                 	print_every_n=5,
                 early_stopping_rounds    = 10,
                 maximize            = FALSE,
                 eval_metric='rmse'
	)

	pred1<-predict(clf,xgb.DMatrix(data.matrix(
					train[,c(8:ncol(train)),with=FALSE]),
					missing=NA))
	pred1.log = expm1( pred1 )
	value1 = kaggle_fun(train$Demanda_uni_equil,pred1.log)

	pred2<-predict(clf,xgb.DMatrix(data.matrix(
					test3[,c(8:ncol(test3)),with=FALSE]),
					missing=NA))
	pred2.log = expm1( pred2 )
	pred2.log[pred2.log<0]=0
	value2 = kaggle_fun(test3$Demanda_uni_equil,pred2.log)
#--------------------------------------------------------------------------
	return( list( c(value1,value2 ) , clf ) )
}

work.train.data.fun=function(test2){

	temp	 = work.train.test.map.fun2( nrow(test2),2000000 )

	train.num=temp[[1]]
	test.num=temp[[2]]

	train=test2[train.num,]
	test3=test2[test.num,]
	return( list(train,test3) )
}
find.map.fun=function(x){
	na.map = apply(x,1,is.na)
	row.na = apply( t(na.map) , 1 , sum )
	#沒有na的data
	map0 = (  row.na == 0 )
	sum(map0)
	return(map0)
}
na.become.0.fun=function(test,i){
	map = is.na( test[,c(i),with=FALSE] )
	test[map[,1],c(i)]<-0
	return(test)
}


work.var.fun=function(main.train.x){

	#main.train.x$log.vuh	= log1p( main.train.x$Venta_uni_hoy )	#sale
	main.train.x$log.vh  = log1p( main.train.x$Venta_hoy )		#sale money
	#main.train.x$log.dup = log1p( main.train.x$Dev_uni_proxima )	#return
	main.train.x$log.due = log1p( main.train.x$Demanda_uni_equil )#adj
	gc()
	#--------------------------------------------------------------------------------------------
	mean.due.age = main.train.x[, .(mean.due.age = mean(log.due)), by = .(Agencia_ID)]
	mean.due.can = main.train.x[, .(mean.due.can = mean(log.due)), by = .(Canal_ID)]
	mean.due.rut = main.train.x[, .(mean.due.rut = mean(log.due)), by = .(Ruta_SAK)]
	mean.due.cli = main.train.x[, .(mean.due.cli = mean(log.due)), by = .(Cliente_ID)]
	mean.due.pro = main.train.x[, .(mean.due.pro = mean(log.due)), by = .(Producto_ID)]
	#--------------------------------------------------------------------------------------------
	mean.vh.age = main.train.x[, .(mean.vh.age = mean(log.vh)), by = .(Agencia_ID)]
	#--------------------------------------------------------------------------------------------
	#--------------------------------------------------------------------------------------------
	mean.due.pa	= main.train.x[, .(mean.due.pa= mean(log.due)), by = .(Producto_ID, Agencia_ID)]
	mean.due.pr	= main.train.x[, .(mean.due.pr= mean(log.due)), by = .(Producto_ID, Ruta_SAK)]
	mean.due.pcli	= main.train.x[, .(mean.due.pcli= mean(log.due)), by = .(Producto_ID, Cliente_ID)]
	mean.due.pcan	= main.train.x[, .(mean.due.pcan= mean(log.due)), by = .(Producto_ID, Canal_ID)]

	mean.due.pca	= main.train.x[, .(mean.due.pca= mean(log.due)), by = .(Producto_ID, Cliente_ID, Agencia_ID)]
	#--------------------------------------------------------------------------------------------

	mean.due.acrcp	= main.train.x[, .(mean.due.acrcp= mean(log.due)), 
		by = .(Agencia_ID , Canal_ID , Ruta_SAK , Cliente_ID , Producto_ID)]

	sd.due.acrcp	= main.train.x[, .(sd.due.acrcp= sd(log.due)), 
		by = .(Agencia_ID , Canal_ID , Ruta_SAK , Cliente_ID , Producto_ID)]

	#--------------------------------------------------------------------------------------------
	return( list(
			mean.due.age,
			mean.due.can,
			mean.due.rut ,
			mean.due.cli ,
			mean.due.pro ,
			mean.due.pa,
			mean.due.pr,
			mean.due.pcli,
			mean.due.pcan,
			mean.due.pca,	
			mean.vh.age,
			mean.due.acrcp,
			sd.due.acrcp
	) )
}

work.model.fun2=function(main.train.y,
                         n,sample.amount,
                         mean.due.age,
                         mean.due.can,
                         mean.due.rut ,
                         mean.due.cli ,
                         mean.due.pro ,
                         mean.due.pa,
                         mean.due.pr,
                         mean.due.pcli,
                         mean.due.pcan,
                         mean.due.pca,
                         mean.vh.age,
                         mean.due.acrcp,
                         sd.due.acrcp){
  
  set.seed(100)
  #n=1000000
  num = sample( c( 1:sample.amount ) ,n )
  
  test = main.train.y[num,.(	Agencia_ID , Canal_ID , Ruta_SAK , 
                             Cliente_ID , Producto_ID , log.due , 
                             Demanda_uni_equil )]	
  #--------------------------------------------------------------------------------------------
  test = merge(test , mean.due.age, all.x = TRUE, by = c("Agencia_ID"))
  test = merge(test , mean.due.can, all.x = TRUE, by = c("Canal_ID"))
  test = merge(test , mean.due.rut, all.x = TRUE, by = c("Ruta_SAK"))
  test = merge(test , mean.due.cli, all.x = TRUE, by = c("Cliente_ID"))
  #test = merge(test , mean.due.pro, all.x = TRUE, by = c("Producto_ID"))
  #--------------------------------------------------------------------------------------------
  #--------------------------------------------------------------------------------------------
  #--------------------------------------------------------------------------------------------
  test = merge(test , mean.due.pa, all.x = TRUE, by = c("Producto_ID", "Agencia_ID"))
  test = merge(test , mean.due.pr, all.x = TRUE, by = c("Producto_ID", "Ruta_SAK"))
  test = merge(test , mean.due.pcli, all.x = TRUE, by = c("Producto_ID", "Cliente_ID"))
  test = merge(test , mean.due.pcan, all.x = TRUE, by = c("Producto_ID", "Canal_ID"))
  #--------------------------------------------------------------------------------------------
  test = merge(test , mean.due.pca, all.x = TRUE, by = c("Producto_ID", "Cliente_ID", "Agencia_ID"))
  #-----------------------------------------------------------------
  test = merge(test , mean.vh.age, all.x = TRUE, by = c("Agencia_ID"))
  
  
  test = merge(test , sd.due.acrcp, all.x = TRUE, 
               by = c("Producto_ID", "Cliente_ID", 
                      "Agencia_ID",	"Canal_ID" , "Ruta_SAK"))
  
  test = merge(test , mean.due.acrcp, all.x = TRUE, 
               by = c("Producto_ID", "Cliente_ID", 
                      "Agencia_ID",	"Canal_ID" , "Ruta_SAK"))
  
  set.seed(100)
  temp = work.train.data.fun( test )
  train=temp[[1]]
  test3=temp[[2]]
  set.seed(100)
  #train,test3,nro,eta=0.1,md=10,cb=0.5
  value = xgb.fun(train,test3,75,0.1,8,0.5)
  (v1=value[[1]])
  
  model1=value[[2]]
  #----------------------------------------------------------------
  return(list(c(v1),model1))
}

bind.var.main.test.fun = 
		function(main.test,
				mean.due.age,
				mean.due.can,
				mean.due.rut ,
				mean.due.cli ,
				mean.due.pro ,
				mean.due.pa,
				mean.due.pr,
				mean.due.pcli,
				mean.due.pcan,
				mean.due.pca,
				mean.vh.age,
				mean.due.acrcp,
				sd.due.acrcp){

	main.test= merge(main.test, mean.due.age, all.x = TRUE, by = c("Agencia_ID"))
	main.test= merge(main.test, mean.due.can, all.x = TRUE, by = c("Canal_ID"))
	main.test= merge(main.test, mean.due.rut, all.x = TRUE, by = c("Ruta_SAK"))
	main.test= merge(main.test, mean.due.cli, all.x = TRUE, by = c("Cliente_ID"))
	#main.test= merge(main.test, mean.due.pro, all.x = TRUE, by = c("Producto_ID"))

	main.test= merge(main.test, mean.due.pa, all.x = TRUE, by = c("Producto_ID", "Agencia_ID"))
	main.test= merge(main.test, mean.due.pr, all.x = TRUE, by = c("Producto_ID", "Ruta_SAK"))
	main.test= merge(main.test, mean.due.pcli, all.x = TRUE, by = c("Producto_ID", "Cliente_ID"))
	main.test= merge(main.test, mean.due.pcan, all.x = TRUE, by = c("Producto_ID", "Canal_ID"))

	main.test= merge(main.test, mean.due.pca, all.x = TRUE, by = c("Producto_ID", "Cliente_ID", "Agencia_ID"))

	main.test= merge(main.test, mean.vh.age, all.x = TRUE, by = c("Agencia_ID"))

	main.test = merge(main.test , sd.due.acrcp, all.x = TRUE, 
		by = c("Producto_ID", "Cliente_ID", 
				"Agencia_ID",	"Canal_ID" , "Ruta_SAK"))

	main.test = merge(main.test , mean.due.acrcp, all.x = TRUE, 
		by = c("Producto_ID", "Cliente_ID", 
				"Agencia_ID",	"Canal_ID" , "Ruta_SAK"))

	return(main.test)
}

work.rfm.data.fun=function(main.train.x){
	#data 排序
	rfm.data0 =
	main.train.x  %>% 
	arrange(	Agencia_ID , Canal_ID , 
			Ruta_SAK , Cliente_ID , 
			Producto_ID , Semana)

	#算 rfm
	temp = summarise(
		group_by(
			rfm.data0, Agencia_ID , Canal_ID , 
			Ruta_SAK , Cliente_ID , Producto_ID) 
	,recent=8-max(Semana),freq=length(Semana),money=mean(Venta_hoy))

	rfm.data = data.table( temp )

	return(rfm.data)
}
freq.level.fun=function(x){
	#x=0
	if(x>=1 && x<=2){
		value=1
	}else if(x>=3 && x<=4){
		value=2			
	}else if(x>=5 && x<=6){
			value=3			
	}
	return(value)
}
recent.level.fun=function(x){
	#x=0
	if(x>=0 && x<=2){
		value=3
	}else if(x>=2 && x<=3){
		value=2			
	}else if(x>=4 && x<=5){
			value=1			
	}
	return(value)
}
money.level.fun=function(x){
	#x=0
	if(x>=0 && x<=19){
		value=1
	}else if(x>19 && x<=36){
		value=2			
	}else if(x>36 ){
			value=3			
	}
	return(value)
}


pred.fun=function(main.train.x,main.train.x.2,train.amount,main.test){
  #--------------------------------------------------------------------------------------------
  #生變數
  main.train.y$log.due	= log1p( main.train.y$Demanda_uni_equil )
  main.train.x = data.table( main.train.x )
  
  temp = work.var.fun(main.train.x)
  mean.due.age	=	temp[[1]]
  mean.due.can	=	temp[[2]]
  mean.due.rut	=	temp[[3]]
  mean.due.cli	=	temp[[4]]
  mean.due.pro	=	temp[[5]]
  mean.due.pa	=	temp[[6]]
  mean.due.pr	=	temp[[7]]
  mean.due.pcli	=	temp[[8]]
  mean.due.pcan	=	temp[[9]]
  mean.due.pca	=	temp[[10]]
  mean.vh.age	=	temp[[11]]
  mean.due.acrcp=	temp[[12]]
  sd.due.acrcp	=	temp[[13]]
  gc()
  #--------------------------------------------------------------------------------------------
  #
  sample.amount = nrow(main.train.y)
  main.train.y = data.table(main.train.y)
  temp2 = work.model.fun2(main.train.y,
                          train.amount,sample.amount,
                          mean.due.age,
                          mean.due.can,
                          mean.due.rut ,
                          mean.due.cli ,
                          mean.due.pro ,
                          mean.due.pa,
                          mean.due.pr,
                          mean.due.pcli,
                          mean.due.pcan,
                          mean.due.pca,
                          mean.vh.age,
                          mean.due.acrcp,
                          sd.due.acrcp)
  gc()
  print( temp2[[1]] )
  model=temp2[[2]]
  
  rm(temp2)
  rm(temp)
  #rm(main.train.x) 	
  gc()
  
  #-----------------------------------------------------------------
  #-----------------------------------------------------------------	
  #-----------------------------------------------------------------
  #模型生好了   接下來是預測
  
  #--------------------------------------------------------------------------------------------
  #生變數
  #main.train.y$log.due	= log1p( main.train.y$Demanda_uni_equil )
  main.train.x.2 = data.table(main.train.x.2)
  temp = work.var.fun(main.train.x.2)
  mean.due.age	=	temp[[1]]
  mean.due.can	=	temp[[2]]
  mean.due.rut	=	temp[[3]]
  mean.due.cli	=	temp[[4]]
  mean.due.pro	=	temp[[5]]
  mean.due.pa	=	temp[[6]]
  mean.due.pr	=	temp[[7]]
  mean.due.pcli	=	temp[[8]]
  mean.due.pcan	=	temp[[9]]
  mean.due.pca	=	temp[[10]]
  mean.vh.age	=	temp[[11]]
  mean.due.acrcp=	temp[[12]]
  sd.due.acrcp	=	temp[[13]]
  rm(temp)
  gc()
  #---------------------------------------------------
  #---------------------------------------------------
  main.test = bind.var.main.test.fun(main.test,
                                     mean.due.age,
                                     mean.due.can,
                                     mean.due.rut ,
                                     mean.due.cli ,
                                     mean.due.pro ,
                                     mean.due.pa,
                                     mean.due.pr,
                                     mean.due.pcli,
                                     mean.due.pcan,
                                     mean.due.pca,
                                     mean.vh.age,
                                     mean.due.acrcp,
                                     sd.due.acrcp)
  
  pred1<-predict(model,xgb.DMatrix(data.matrix(
    main.test[,c(8:ncol(main.test)),with=FALSE]),
    missing=NA))
  pred1.log = expm1( pred1 )
  pred1.log=as.integer(round(pred1.log))
  
  result1 = data.table(	id=main.test$id,
                        Demanda_uni_equil=pred1.log)
  
  return((result1))
}



















