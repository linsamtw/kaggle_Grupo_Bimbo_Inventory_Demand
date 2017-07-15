
# kaggle Grupo Bimbo Inventory Demand, NO.156/top 8%
 [Grupo Bimbo Inventory Demand](https://www.kaggle.com/c/grupo-bimbo-inventory-demand)<br>
 e-mail : samlin266118@gmail.com <br>
 如果有問題可以直接寄信給我 <br>
 **********************************************
 結論 : 
 
 注意 : 
 **********************************************
 # 1. 緒論
 
 此問題是關於一家位於墨西哥的連鎖麵包店 --- 庫存需求，由於麵包過期所造成的損失，
 估計一個星期約兩千萬台幣。這不單單是金錢上的損失，更是資源上的浪費。
 除此之外，庫存控管問題，不只發生在這家企業，只要有進行銷售行為的產業，
 都會有相關的問題，因此我選擇此問題進行分析預測。
 
 # 2. 資料介紹
 Grupo Bimbo Inventory Demand 位於墨西哥，一百多萬家商店，以下是是該連鎖店一部分的分布圖
 
 ![google map](https://github.com/f496328mm/kaggle_Grupo_Bimbo_Inventory_Demand/blob/master/bimbo.jpg)
 
 部分店家位於郊區，部分商店位於市區，由不同的路線運送麵包，郊區與市區，對於麵包的需求也不相同，這會是我們考慮的變數之一。
 
 詳細變數意義，可以參考 [kaggle](https://www.kaggle.com/c/grupo-bimbo-inventory-demand/data)的敘述，
 我使用的變數大約是以下這些
 
 |variable|意義|
 |--------|---|
 |Demanda_uni_equil|Adjusted Demand (integer) (Target)|
 |Semana|Week number|
 |Agencia ID|Sales Depot ID|
 |Canal ID | Sales Channel ID|
 |Producto ID | Product ID|
 |Ruta SAK | Route ID (Several routes = Sales Depot)|
 |Cliente ID | Client ID|
 |Venta uni hoy | Sales unit this week (integer)|
 |Dev uni proxima|Returns unit next week (integer)|
 
 根據我們的variable selection，其他變數相對不重要，並沒有放入model中，
 詳細的variable selection將會在後面介紹，我們的目標不單單是最小化庫存損失，
 而是提高利潤。也就是說，過於低估會降低銷售量(庫存0就不會損失，然後就倒閉了QQ)，並不是我們希望的。
 
 而評估準則是 [Root Mean Squared Logarithmic Error(RMSLE)]()
 
<img src="http://chart.googleapis.com/chart?cht=tx&chl=\Large x=\frac{-b\pm\sqrt{b^2-4ac}}{2a}" style="border:none;">

 ### 2.1 資料準備 
 
# 3. 特徵製造
### 3.1 feature engineering 1 ( 特徵工程 1 )



### 3.2 feature engineering 2 ( 特徵工程 2 )



### 3.3 變數選擇


### 3.4 feature selection

### 3.5 other 


   
# 4. Fitted model





# 50 feature
  

# Reference

 [Bosch Production Line Performance. ( 2016 ) ](https://www.kaggle.com/c/bosch-production-line-performance )<br>

[Daniel FG. ( 2016 )](https://www.kaggle.com/danielfg/xgboost-reg-linear-lb-0-485)







 



