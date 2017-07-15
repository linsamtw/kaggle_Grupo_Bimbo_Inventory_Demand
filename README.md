
# kaggle Grupo Bimbo Inventory Demand, NO.156/top 8%
 [Grupo Bimbo Inventory Demand](https://www.kaggle.com/c/grupo-bimbo-inventory-demand)<br>
 e-mail : samlin266118@gmail.com <br>
 如果有問題可以直接寄信給我 <br>
 **********************************************
 這是我初學的第一個比賽，由於該比賽已經過期，因此排名是我藉由提交預測結果，
 kaggle計算出的得分，回推的排名，並非真實比賽。
 
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
 
 
 而評估準則是 [Root Mean Squared Logarithmic Error(RMSLE)](https://www.kaggle.com/c/grupo-bimbo-inventory-demand#evaluation)
 
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
 

# 3. 特徵工程
在這步驟，根據我目前的經驗，這是ML最重要的一環，除非是自己寫 model，
不然用的model與其他人不會有差別，每個人都會用，那憑什麼做的比其他人好?
因此重點就在於 feature ，根據 [kaggle ceo](https://www.import.io/post/how-to-win-a-kaggle-competition/)的文章，
the secret to winning Kaggle competitions，有兩個方法，其中一個就是Handcrafted feature engineering，
因此將介紹我們在這個問題上，使用的feature engineering。


8個類別變數，5個數值變數，數值變數主要是，
該產品 sales、sales 金額、return、return 金額與 Demanda_uni_equil，
而 sales、return 與 Demanda_uni_equil 幾乎是線性的，sales - return = Demanda_uni_equil。
有一點需要注意，Demanda_uni_equil 數據過於偏右，mean(7.225) 大於 Q3(6)，
因此對該變數取log，將此變數分布往中間集中。


除了數值變數之外，我們的 feature engineering 主要是對於類別變數進行處理。
我們不使用一般傳統的方法( indicator matrix )，而是使用另一種方法，
對於該"類別"，在"目標變數"上過去的平均表現，取代該類別。
舉例來說，"紅豆麵包過去平均庫存需求量"，"商店A過去平均庫存需求量"，"路線B過去平均庫存需求量"等等，
將 "紅豆麵包"、"商店A"、"路線B" 這些類別，用 "過去平均庫存需求量" 取代，轉換為數字，而數字我們也比較容易處理。


參考 code 如下： ( due = Demanda_uni_equil，log.due = log( Demanda_uni_equil ) )<br>
mean.due.product = train_data[,.(mean.due.product = mean(log.due)),by=c("product_id")]


以上是不同產品過去的平均表現，對 log.due 取 mean。
也可以同時對兩種類別變數做 feature engineering ，也就是 "紅豆麵包" 在 "商店A" 過去平均庫存需求量，
藉由以上方法，我們製造非常多feature，接下來就是feature selection。

### 3.3 變數選擇
在資料分析上，feature selection往往是最後在做的事，我們先盡可能製造各種變數，再來做feature selection。
當你變數不夠多時，做feature selection是沒意義的。

### 3.4 feature selection
我們使用的方法是 forward selection，藉由XGBoost model計算error，觀察加入變數前後，
testing error有無下降，作為評斷標準。在 feature engineering 上，
選擇 forward selection 是直覺的，因為我們不可能一開始就把所有的feature都製造出來，
過程應該是，一步一步找出feature，不斷製造各種不同的變數，
我們無法事前得知哪些變數重要，只能利用經驗與視覺化分析，協助找出比較有可能的feature。




### 3.5 other 


   
# 4. Fitted model





# 延伸討論
  我並沒有使用時間序列上，lag term 作為變數，未來可以往這個方向去加強模型。

# Reference

 [Bosch Production Line Performance. ( 2016 ) ](https://www.kaggle.com/c/bosch-production-line-performance )<br>

[Daniel FG. ( 2016 )](https://www.kaggle.com/danielfg/xgboost-reg-linear-lb-0-485)







 



