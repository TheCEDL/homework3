# Homework3 - Generative Models 
This is a team homework.<br>
In your report please list the contribution of each team member.

# How to complete the task

本次作業目標為採用DCGAN的tensorflow版本去生成256乘256解析度的室內場景輸出圖片，原本DCGAN的generator是利用deconvolution架構stride為2的原版本的架構，generator每增加的層數會比前一層放大2倍，藉由此方式來一層一層放大產生feature map並達到相對應的輸出結果，而我們的實作架構如下圖:

<img src=/image/1.png width=800 />

如上圖所示，為了達成目的我們在generator增加2層的deconvolutional layer來放大featuremap的解析度，而discriminator則為generator架構的鏡像反射，其有相對應的convolutional layer並在訓練時和generator進行拉鋸戰學習。

除了增加Deconv的層數的架構讓output size變大之外，額外的我們嘗試不同kernerl size 對輸出結果有何影響，我們想以這個變量因素去做實驗的原因為想解決輸出圖片棋盤化(artifact)的問題，由於deconvolutional layer來放大時kernel的stride具有“不均勻的重疊”問題如下圖所示，這會導致某些產出的pixel會被重複計算並偏向不理想的值。
<img src=/image/2.png width=800 />

由於一開始的版本kernel的size為5，stride為2會產生很嚴重的pixel間隔一亮一暗的棋盤化(artifact)效應，如下圖所示
<img src=/image/3.png width=800 />

因此適當的調整kernel的大小改為size為4或6時（也就是說當kernel size可以被stride整除時），便能夠讓輸出空間上的棋盤化(artifact)效應降低，這也就是我們想嘗試的目標，示意圖如下<br>
<img src=/image/4.png width=800 /> 
<img src=/image/5.png width=800 />

此外我們也嘗試了改變輸入雜訊z的維度，觀察這一項改變是否會對生成圖片或是訓練過程有所改變。

# Analysis of result

在generator中增加2層的deconvolutional layer來放大到256x256 output size，並使用100維度的noise vector、kernel的size為5、stride為2（皆與原本source code相同），第12個epoch的結果如下圖所示：
<img src=/image/6.png width=800 /> 
很明顯的可以看出，整個架構無法完整的產生室內景致的圖像，我們推測的原因可能是一開始noise vector投影到的第一層feature map解析度太小(4x4@1024)，並且100維度的noise vector可能不足以表達如此複雜的室內景色。因此，我們嘗試把noise vector的維度改為1600，並且投影到8x8@1024的feature map之上，其他條件不變，第12個epoch的結果如下圖所示:
<img src=/image/7.png width=800 /> 

此外為了解決棋盤效應的問題，我們也將了每一層deonv的kernel size改為6，使得其為stride 2之倍數，並使用1600維的noise vector投影到8x8@512 feature map，以下是24 epoch的結果：
<img src=/image/8.png width=800 /> 

可以很明顯的看出上面白色缺口的影響變的比較部會那麼的嚴重，然而我們法線圖片細緻的程度卻不及於使用kernel size為6的結果，最後我們嘗試把noise vector擴大，觀察其輸出的變化，我們將3200維度的noise vector投影到8乘8深度為512的feature map之上，其kernel的size為5，stride為2第16個epoch的結果如下圖所示：
<img src=/image/9.png width=800 /> 
我們觀察出擴大noise vector其會影響generator整體影像色調很大的幅度，其會讓所有training的sample整體變成統一的色調顏色。

# Conclusion
1. Noise vector 一開始投影到越高維度的feature map上，其輸出效果越好
2. Kernel size的size和stride會影響白色缺口棋盤化(artifact)的嚴重程度，Kernel size愈低缺口愈凝聚並愈小，但產生的圖片卻會愈模糊
3. noise vector的大小會影響generator訓練時的色調變化程度，noise vector愈大變化愈劇烈
最終我們使用 6x6 Deconv/ 1600維 noise vector/ 投影第一層conv為 8x8@512

# 500 generated data
<a>https://drive.google.com/file/d/0B5LE_6igFDn_NHRIVllzc1prVEU/view?usp=sharing</a>

## Participation
| Name | Do |
| :---: | :---: |
| 郭士鈞 | 實作實驗、撰寫報告 |
| 黃冠諭 | 算法討論、撰修報告 |
| 蘇翁台 | 算法討論、撰修報告 |
