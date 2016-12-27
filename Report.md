# CEDL Homework3 - Generative Models


  本次的作業要使用DCGAN來產生室內場景的影像，GAN主要由一個Generator和一個Discriminator構成，Generator要不斷地去欺騙Discriminator，希望Discriminator辨識它產生的影像或其他數據為真實數據，而Discriminator得去分辨輸入資料的真實性，並在訓練中不斷視Generator的資料為假的資料，經由這一來一往的過程，從而訓練整個網路。
  
  
  DCGAN的貢獻似乎在於將一些目前的技術，結合到了GAN中，他用strided convolutions來進行卷積，用Fractional-Strided Convolutions進行反卷積，全程不用pooling，並加入了Batch Normalization使網路更加穩定。
  

## Modifications

我們在架構和方法上沒有做太多的變化，主要都只是參數調整上的改變。

### 嘗試一

  1. 和提供的原始程式架構無異（一線性轉換層、四卷積層）
  
  2. 將輸出調整為256，並加長訓練的epoch為30，因為要輸出比原程式更大的圖，或許需要更長的時間來達到收斂。
  
### 嘗試二 

  1. Generator和Discriminator各多加一層（一線性轉換層、五卷積層）
  
  2. 起始feature map的數量為1024，到第五層卷基層時剩64，感覺比較多的feature map會畫出比較精確的結果。
  
  3. 將z的維度加長到400維，我們認為或許可以增加輸出影像的多樣性，但又擔心維度太多的z，訓練的model無真的產出多樣的影像，不過第2點和第3點並沒有時間，訓練更多的model來驗證。
  
## Experiment Discusion
### 結果一

  <img src=/image/1.png width=600 />

### 結果二

  <img src=/image/2.png width=600 />
  
  從兩個嘗試的結果來看，結果一上具有許多空白的小點，呈現的結果也稍微比較混亂，推測因為feature map數量不夠多，原始DCGAN的paper要生成64x64的影像，初始的feature map數量為1024，但由於程式碼中的資料以人臉和數字為主，相比於場景是比較沒那麼複雜的資料，故將feature map數量降為一半，這樣的設置或許不適用於場景影像的生成。
  
  結果二若單從比較成功的影像來看，呈現的影像比較乾淨和工整，不過其中有幾張卻有巨大的空白，沒有產出完整的區塊，也有比較多整張異色的影象，又由於許多結果有一些尚未收斂時的那種紋路，推測訓練的epoch（25），似乎還沒辦法讓整個model收斂完畢，又或著z的維度其實不太需要調整，應該要調降。從綜合結果來看，效果並未大大提升，但感覺是一個方向。
  
## Evaluation Method
利用嘗試一的model去discriminate提供的H3eval dataset，得到結果如下圖所示
<img src=/image/3.png width=300 />

之後我們用-5的threshold去辨別是不是真實室內場景，可得到1或0的字串。 
  
  
  
  
