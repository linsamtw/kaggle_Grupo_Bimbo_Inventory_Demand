
# kaggle Grupo Bimbo Inventory Demand, NO.156/top 8%
 [Grupo Bimbo Inventory Demand](https://www.kaggle.com/c/grupo-bimbo-inventory-demand)<br>
 e-mail : samlin266118@gmail.com <br>
 如果有問題可以直接寄信給我 <br>
 **********************************************
 這是我初學的第一個比賽，由於該比賽已經過期，因此排名是我藉由提交預測結果，
 kaggle計算出的得分，回推的排名，並非真實比賽。不過我的方法優於大部分參賽者提出的kernel。
 
 注意 : 
 **********************************************
 # 1. 緒論
 
 此問題是關於一家位於墨西哥的連鎖麵包店 --- 庫存需求，由於麵包過期，造成一個星期約兩千萬台幣的損失。
 這不單單是金錢上的損失，更是資源上的浪費。
 除此之外，庫存控管問題，不只發生在這家企業，只要有進行銷售行為的產業，
 都會有相關的問題，因此我選擇此問題進行分析預測。
 
 # 2. 資料介紹
 Grupo Bimbo 位於墨西哥，一百多萬家商店，以下是是該連鎖店一部分的分布圖
 
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
 而是提高利潤。也就是說，過於低估會降低銷售量( 庫存0就不會損失，然後就倒閉了QQ )，並不是我們希望的。
 
 
 而評估準則是 [Root Mean Squared Logarithmic Error(RMSLE)](https://www.kaggle.com/c/grupo-bimbo-inventory-demand#evaluation)
 
 ### 2.1 資料準備 
 
|data|size|n (資料筆數)|p (變數數量)| 時間長度 |在 R 中佔的 ram |
|----|----|-----------|------------|------|---------------|
|training data|3 GB|74 million 筆|8個類別變數，5個數值變數| week 3 ~ week 9| 3.6 GB |
|testing data|239 MB|7 million 筆|8個類別變數，5個數值變數|week 10 ~ week 11| 0.2 GB |
 
 training data 大小約3GB，7個星期的相關資料，8千萬筆data，8個類別變數，5個數值變數。
 資料中並沒有標示日期，只有 week 3 ~ week 9，無法得知時間點。
 testing data 是關於 week 10 ~ week 11 的庫存需求，這是我們要預測的目標。
 由於是預測未來，跟一般的ML不同，因此將進行資料切割。<br>
 
 ### 2.2 資料切割
 我們只有過去 7 個星期的資料，要預測未來 2 個星期不太容易，因此先簡化問題為，
 未來 1 個星期，將 testing data 中的 week 11 視為 week 10。<br>
 
 
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
重點就在於 feature ，根據 [kaggle ceo](https://www.import.io/post/how-to-win-a-kaggle-competition/)的文章，
the secret to winning Kaggle competitions，有兩個方法，其中一個就是Handcrafted feature engineering，
因此將介紹在這個問題上，使用的feature engineering。<br><br>

### 3.1 feature engineering
8個類別變數，5個數值變數，有一點需要注意，Demanda_uni_equil (庫存需求) 過於偏右，mean(7.225) 大於 Q3(6)，
因此對該變數取log，將此變數分布往中間集中。<br><br>


除了數值變數之外，我們的 feature engineering 主要是對於類別變數進行處理。
不使用一般傳統的方法( indicator matrix )，而是使用另一種方法，
對於該"類別"，在"目標變數"上過去的平均表現，取代該類別。<br><br>

舉例來說，"紅豆麵包過去平均庫存需求量"，"商店A過去平均庫存需求量"，"路線B過去平均庫存需求量"等等，
將 "紅豆麵包"、"商店A"、"路線B" 這些類別，用 "過去平均庫存需求量" 取代，轉換為數字，而數字我們也比較容易處理。<br><br>


參考 code 如下： ( due = Demanda_uni_equil，log.due = log( Demanda_uni_equil ) )<br>
mean.due.product = train_data[,.(mean.due.product = mean(log.due)),by=c("product_id")]<br><br>


以上是不同產品過去的平均表現，對 log.due 取 mean。
也可以同時對兩種類別變數做 feature engineering ，也就是 "紅豆麵包" 在 "商店A" 過去平均庫存需求量，
藉由以上方法，我們製造非常多feature，接下來就是feature selection。

### 3.2 feature selection
在資料分析上，feature selection往往是最後在做的事，我們先盡可能製造各種變數，再來做feature selection。
當你變數不夠多時，做feature selection是沒意義的。<br><br>

我使用的方法是 forward selection，藉由XGBoost model計算error，觀察加入變數前後，
testing error有無下降，作為評斷標準。<br><br>

在 feature engineering 上，
選擇 forward selection 是直覺的，因為我們不可能一開始就把所有的 feature 都製造出來，
過程應該是，一步一步找出 feature，不斷製造各種不同的變數，
我們無法事前得知哪些變數重要，只能利用經驗與視覺化分析，協助找出比較有可能的feature。


對於初學者來說，需要特別注意，將 seed 設定好，否則每次結果不同，無法保證該變數是 feature or noise。

以下是我加入變數後，error 下降的過程

|Add Feature|The Feature Meaning|RMSLE of Train|RMSLE of Test|
|-----------|-------------------|--------------|-------------|
|baseline |-| 0.718| 0.728|
|+mean.due.pa|the mean of log.due with Producto ID and Agencia ID.|0.525|0.536|
|+mean.due.pr|the mean of log.due with Producto ID and Ruta SAK.|0.511|0.525|
|+mean.due.pcli|the mean of log.due with Producto ID and Cliente ID.|0.455|0.467|
|+mean.due.pcan|the mean of log.due with Producto ID and Canal ID.|0.449|0.462|
|+mean.due.pca|the mean of log.due with Producto ID, Cliente ID and Agencia ID.|0.449|0.461|
|+mean.vh.age|it is mean of nature log Venta hoy with Agencia ID.|0.449|0.461|
|+sd.due.acrcp|it is standard deviation of log.due with Producto ID, Cliente ID, Agencia ID, Canal ID and Ruta SAK|0.446|0.460|
|+mean.due.acrcp|the mean of log.due with Producto ID, Cliente ID,|0.445|0.459|

經過 feature engineering ，整體上進步約40%( 0.728 -> 0.459 )

baseline 是使用 mean.due.Agencia_ID, mean.due.Canal_ID, mean.due.Ruta_SAK, mean.due.Cliente_ID 這些變數，
可以，很明顯看出，在加入變數後，testing error 逐步下降，而實際上我們進行非常多次的 feature engineering，
最後的結果看似很簡短，實際上需要花非常多時間。

# 4. MODEL

我選用 XGBoost ，做為我們的model，這是一個 tree & GB 的model。在大多數Kaggle問題中，
基本上都是 XGBoost or DL，XGB 速度上非常快，
相較於其他的ML model(SVM, RF, TREE)，他使用多核心計算，所以速度上快上不少。<br><br>

實際上 XGBoost 可以比 RF 快上100倍，為何產生這樣的差異，
該XGB的作者 - [Tianqi Chen](https://www.quora.com/What-makes-xgboost-run-much-faster-than-many-other-implementations-of-gradient-boosting)給出了更詳細回應，詳細可以參考 [XGBoost paper](https://arxiv.org/abs/1603.02754)，在這不多做解釋。<br><br>

另外XGB可以藉由 sparse matrices 進行建模，在實際問題上，missing value是一定會發生的，
因此這個優勢也是我們選擇它的原因之一。它處理 sparse matrices 的方法，要回到tree的概念，
一般tree就是做個二分法，也就是說，即使遇到NA，你也可以試著將它分到 左 or 右 ，再來計算loss function，
簡單來說就是做個猜測，選一個最好的方向，minimise loss function。
藉由這種想法，再利用algorithm去優化它，進而處理 sparse matrices ，
詳細內容可以參考 [XGBoost paper](https://arxiv.org/abs/1603.02754)。
   
# 5. Fitted model
最後的模型，可以利用以下迴歸的方法表示：

|y| |x|
|-|-|-|
|log.due| ~ |mean.due.Agencia ID |
|||mean.due.Agencia ID| 
|||mean.due.Canal ID |
|||mean.due.Ruta SAK |
|||mean.due.Cliente ID| 
|||mean.due.pa |
|||mean.due.pr |
|||mean.due.pcli |
|||mean.due.pcan |
|||mean.due.pca |
|||mean.vh.age |
|||sd.due.acrcp |
|||mean.due.acrcp|

# 6. 結論
該篇主要重點可以分為以下兩點：
1. feature engineering，主要是利用過去平均表現，取代類別變數。<br>

2. XGBoost 優於一般ML( SVM,RF,TREE,GB... )，實際問題上，
data超過10GB、100GB是正常的，因此快速的建模是必要條件，抽樣則會捨棄大數據的威力。<br>

3. 該篇分析方法適不適用其他問題？如果單純只有這次有用，那沒有讀的價值，
   所以我對另外一個類似的比賽---[Rossmann Store Sales](https://github.com/f496328mm/kaggle_Rossmann_Store_Sales) 進行分析，
   有關銷售量預測，我認為銷售量與庫存量是類似的，使用相同方法做為出發，最後也得到不錯的成績(top 10% rank)，也就是說，未來遇到類似問題，
   該篇方法可以做為一個出發點。<br>
   
4. 一致化的建模，而非將問題拆解，拆解會使問題複雜化，一致化解決問題，將使問題簡單。
   了解model背後的原理，有助於建模，至少要了解參數意義，
   而非都使用預設值( 研討會看到蠻多碩士生都不懂自己的model，甚至是教授 )。


# 延伸討論
  1. 我並沒有使用時間序列上，lag term 作為變數，未來可以往這個方向去加強模型，
     [kernel](https://www.kaggle.com/bpavlyshenko/bimbo-xgboost-r-script-lb-0-457)。<br><br>
  2. 此問題可以延伸到3C產品上，當供應商手上有1000隻IPHONE，要如何分配，才能達到最大獲利?<br>
     舉例來說，如果給A商店900隻手機，B商店100隻手機，但是A商店需求量是500隻，B商店需求量是500隻，<br>
     這將增加A商店的倉儲成本，而B商店由於客戶無IPHONE購買，可能轉往購買其他廠牌，導致IPHONE市佔率下降。<br>
     而更遠的層面為，國家的銷售，不同國家市佔率、關稅、運費等因素不同，我們希望：

     
 |1|最高市佔率，在市佔率低的地區，不希望造成缺貨的問題|
 |-|---------------------------------------------|
 
 |2|同樣價格上，在最低關稅的國家進行銷售，將使獲利提升|
 |-|-----------------------------|
 
 |3|新產品推出時間點早於競爭對手，可能會搶到市佔率，如果利用空運，那路線上的安排，運費是要考量的因素之一|
 |-|-----------------------------|

# Reference

[kernel - Paulo Pinto. ( 2016 )](https://www.kaggle.com/paulorzp/log-mean-plus-lb-0-47000/code)







 



