# Homework3 - Generative Models 
This is a team homework.<br>
In your report please list the contribution of each team member.

# How to complete the task

本次作業目標為採用DCGAN的tensorflow版本去生成256乘256解析度的室內場景輸出圖片，原本DCGAN的generator是利用deconvolution架構stride為2的原版本的架構，generator每增加的層數會比前一層放大2倍，藉由此方式來一層一層放大產生feature map並達到相對應的輸出結果，而我們的實作架構如下圖:

<img src=/image/1.png width=800 />

如上圖所示，為了達成目的我們在generator增加2層的deconvolutional layer來放大featuremap的解析度，而discriminator則為generator架構的鏡像反射，其有相對應的convolutional layer並在訓練時和generator進行拉鋸戰學習，

# Detail of implement

除了嘗試直觀的架構之外，額外的我們嘗試不同kernerl size 對輸出結果有何影響，我們想以這個變量因素去做實驗的原因為想解決輸出圖片棋盤化(artifact)的問題，由於deconvolutional layer來放大時kernel的stride具有“不均勻的重疊”問題如下圖所示，這會導致某些產出的pixel會被重複計算並偏向不理想的值，

<img src=/image/2.png width=800 />

由於一開始的版本kernel的size為5，stride為2會產生很嚴重的pixel間隔一亮一暗的棋盤化(artifact)效應，如下圖所示
<img src=/image/3.png width=800 />

因此適當的調整kernel的大小改為size為4或6時（也就是說當kernel size可以被stride整除時），便能夠讓輸出空間上的棋盤化(artifact)效應降低，這也就是我們想嘗試的目標，如下圖所示<br>
<img src=/image/4.png width=800 /> 可以看到
<img src=/image/5.png width=800 />


# Analysis of result

# 500 generated data
<a>https://drive.google.com/file/d/0B5LE_6igFDn_NHRIVllzc1prVEU/view?usp=sharing</a>

## Participation
| Name | Do |
| :---: | :---: |
| 郭士鈞 | 實作實驗、撰寫報告 |
| 黃冠諭 | 算法討論、撰修報告 |
| 蘇翁台 | 算法討論、撰修報告 |
