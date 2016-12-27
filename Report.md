# CEDL Homework3 - Generative Models


  本次的作業要使用DCGAN來產生室內場景的影像，GAN主要由一個Generator和一個Discriminator構成，Generator要不斷地去欺騙Discriminator，希望Discriminator辨識它產生的影像或其他數據為真實數據，而Discriminator得去分辨輸入資料的真實性，並在訓練中不斷視Generator的資料為假的資料，經由這一來一往的過程，從而訓練整個網路。
  
  
  DCGAN的貢獻似乎在於將一些目前的技術，結合到了GAN中，他用strided convolutions來進行卷積，用Fractional-Strided Convolutions進行反卷積，全程不用pooling，並加入了Batch Normalization使網路更加穩。
  

## Modifications

我們在架構和方法上沒有做太多的變化

### 嘗試一
  和提供的原始程式架構無異（一線性轉換層、四卷積層），但將輸出調整為256，並加長訓練的epoch為30
### 嘗試二 
  Generator和Discriminator各多加一層（一線性轉換層、五卷積層），起始feature map的數量為1024，到第五層卷基層時
## Experiment Discusion
### 

