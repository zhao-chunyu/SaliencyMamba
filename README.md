# <img src="fig\logo.png" alt="image-20240726153118844" style="zoom: 7%;" />SalM$^2$: Saliency Mamba Model

 - *For our proposed Sal method, we collect datasets and other popular modeling codes.*
 - *We give a series of instructions and demo files.*
 - *We promise to give the complete code and result files after the paper is accepted.*

## 🔥Update



## 💬Motivation



## ⚡Proposed Model

<img src="fig\model_overview.png" alt="image-20240726153118844" style="zoom: 10%;" /> 

## 📖Datasets

| Name         | Train  | Valid | Test  | Dataset example                                              |
| ------------ | ------ | ----- | ----- | ------------------------------------------------------------ |
| TrafficGaze  | 49080  | 6655  | 19135 | <img src="fig\TrafficGaze_example.gif" alt="BDDA-3" style="zoom:10%;" /> |
| DrFixD-rainy | 52291  | 9816  | 19154 | <img src="fig\DrFixD-rainy_example.gif" alt="BDDA-1" style="zoom:10%;" /> |
| BDDA         | 286251 | 63036 | 93260 | <img src="fig\BDDA_example.gif" alt="BDDA-0" style="zoom:40%;" /> |

​	***For all datasets we will provide our download link with the official link. Please choose according to your needs**.

> (1) **TrafficGaze**: This dataset we uploaded in [link](www.baidu.com "Download Traffic_Gaze"). We crop 5 frames before and after each video. Official web in [link](https://github.com/taodeng/CDNN-traffic-saliency "Official Traffic_Gaze")。
>
> (2) **DrFixD-rainy**: This dataset we uploaded in [link](www.baidu.com "Download DrFixD-rainy"). We crop 5 frames before and after each video. Official web in [link](https://github.com/taodeng/DrFixD-rainy "Official DrFixD-rainy")。
>
> (3) **BDDA**: This dataset we uploaded in [link](www.baidu.com "Download BDDA"). Some camera videos and gazemap videos frame rate inconsistency, we have matched and cropped them. Some camera videos do not correspond to gazemap videos, we have filtered them. Official web in [link](www.baidu.com "Official BDDA")。

## 🛠️ Deployment **[🔁](#Motivation)**

------

### 	Run train 

​	如果您希望将我们的模型重新训练，请采用以下步骤进行。

1. Train our model, 使用--dataset切换数据集，数据集包括Traffic_Gaze、DrFixD-rainy、BDDA

```python
python main.py --dataset 'BDDA'
```

​	2. Train compare model, 如果模型是静态预测方法，命令如下

```python
python main.py --dataset 'Traffic_Gaze'
```

​	3. Train compare model, 如果模型是动态预测方法，命令如下

```python
python main.py --dataset 'DrFixD-rainy'
```

### 	Run test 

​	如果您希望直接使用我们的模型结果进行对比，您可以采用以下步骤进行。

1. Test our model, 使用--dataset切换数据集，数据集包括Traffic_Gaze、DrFixD-rainy、BDDA

```python
python test.py --dataset 'BDDA'
```

​	2. 如果您由于一些其他原因无法对环境进行适配，您也可以直接下载我们的预测结果。

## 🚀 Live Demo **[🔁](#Motivation)**

------

<img src="fig/model_prediction1.gif" alt="BDDA-1" style="zoom:25%;" /><img src="fig/model_prediction2.gif" alt="BDDA-2" style="zoom:25%;" />

## ✨ Downstream Task

- ***Saliency object detection***: saliency map → **Guide** → object detection

- ***Event recognition***: saliency map → **Guide** → event recognition

## ⭐️ Cite **[🔁](#Motivation)**

If you find this repository useful, please use the following BibTeX entry for citation.

```python
wait accepted
```
