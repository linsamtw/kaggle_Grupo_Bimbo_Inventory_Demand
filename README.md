
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
 
 ### 2.1 資料準備 
 
 training data 大小約3GB，7個星期的相關資料，8千萬筆data，8個類別變數，5個數值變數。
 資料中並沒有標示日期，只有 week 3 ~ week 9，無法得知時間點。
 testing data 是關於 week 10 ~ week 11 的庫存需求，這是我們要預測的目標。
 由於是預測未來，跟一般的ML不同，因此將進行資料切割。
 
 ### 2.2 資料切割
 我們只有過去 7 個星期的資料，要預測未來 2 個星期不太容易，因此先簡化問題為，
 未來 1 個星期，將 testing data 中的 week 11 視為 week 10。
 
 |預測未來 2 周|Week|Week|
 |------------|----|-|
 |真實情況，要預測 week 10 與 11 的庫存需求|3~9|10~11|
 ||training data|testing data|
 |假設 week 11 的 y 與 week 10 相同|3~9|10|
 ||training data|testing data|
 | 建立模型，y 是假的 testing data|3~8|9|
 ||x|y|
 |最後預測，時間進行平移|4~9|10|
 ||x|y|
 
 因為要預測未來，因此進行以上調整，與一般cross validation不太相同。
 

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







 



