組員: 張嘉哲、林暘峻
這一次作業是利用DCGAN去訓練place2的dataset生成假資料提交500張上去，再利用我們所得到的model去discriminate H3eval dataset <br>
首先先利用place2的dataset訓練生成照片。 <br>
生成照片 <br>
![image](https://github.com/chang810249/homework3/blob/master/train_30_0769.png) <br>
提交之後再利用我們的model去discriminate H3eval dataset，可以得到如下列數字 <br>
透過discriminator <br>
![image](https://github.com/chang810249/homework3/blob/master/D_digits.PNG) <br>
之後我們用的threshold是-5去辨別是不是真實室內場景，可得到如下圖類似的字串。 <br>
threshold為-5辨識後 <br>
![image](https://github.com/chang810249/homework3/blob/master/dis.PNG) <br>
