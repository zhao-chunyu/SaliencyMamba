# Deployment Details

### (1) Environment

Due to the fact that the configuration of Mamba is not simple, our configuration process is as follows.

* Git clone this repository

```python
git clone https://github.com/zhao-chunyu/SaliencyMamba
cd SaliencyMamba
```
* Create conda environment

```python
conda create -n salmamba python=3.10
conda activate salmamba
```
* Install PyTorch 2.2.0+cu121

```python
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
```

* Install `causal_conv1d` and `mamba`

```python
pip install -r requirements.txt
pip install -e utils/models/causal-conv1d
pip install -e utils/models/mamba
```

> [note] For the configuration of the Mamba environment, we refer to the configuration process of [VideoMamba](https://github.com/OpenGVLab/VideoMamba/tree/main/videomamba). 

> [note] Please ensure that the driver of your `graphics card` and the version of `CUDA` are strictly matched.

> [note] The configuration of the environment might be complicated. Please be patient and wish you success.

### (2) Run train

​	👉 *If you wish to train with our model, please use the proceeding steps below.*

1. Train our model.  Command: `python train.py --network salmm --b 32 --g 0 --category xxx --root xxx`. You can use `--category` to switch datasets, which include `TrafficGaze`, `DrFixD-rainy`, `BDDA`. `--b` sets batch size, `--g` sets id of cuda. 
	**An official running code :**
```python
python train.py --network salmm --b 32 --g 0 --category TrafficGaze --root dataset/TrafficGaze
```

2. Train the compare model (<u>static prediction method</u>). Command: `python train.py --network xxx --b 32 --g 0 --category xxx --root xxx`. 
	**An official running code :**

```python
python train.py --network cdnn --b 32 --g 1 --category TrafficGaze --root dataset/TrafficGaze
```

3. Train the compare model (<u>dynamic prediction method</u>). Command: `python train.py --network xxx --b 32 --g 0 --seq_len 6 --category xxx --root xxx`. You can set `--seq_len` and run the following command.
	**An official running code :**

```python
python train.py --network DrFixD_rainy --b 32 --g 2 --seq_len 6 --category TrafficGaze --root dataset/TrafficGaze
```

### (3) Run Test

#### 		[1] Official test [⭐⭐⭐]

we calculate the predicted values and then use `Matlab` for the prediction. [More details](metrics/README.md)

```python
cd metrics
chmod +x run_matlab.sh
```

#### 		[2] General test

Although `Python` testing is more convenient, our test benchmark is based on the previous work (`CDNN`、`DrFixD-rainy`、......), and the results calculated by `Python` do not match those calculated by `Matlab`. We have provided a `Python` test code, which is basically consistent with `Matlab` in terms of `CC`, `SIM`, and `KLD` metrics.

​	👉 *If you have completed training a model, you can test using the following code. Here, `--test_weight` refers to a folder where the weights are saved, similar to `salmm_20250304-15:46:29.`*

​	1. Test your model.

```python
python evaluate_metrics.py --network salmm --b 1 --g 0 --category xxx --root xxx --test_weight xxx
```

> [**note**📕] Our final test still adopts Matlab. This is because Matlab was used for calculation in the previous work (`CDNN`、`DrFixD-rainy`、......). 

> [**note**📕] The metrics calculated here using `Python` only include three `distribution-based` metrics `CC`, `SIM`, and `KLD`. **(Python ≈ Matlab)**

> [**note**📕] The `location-based` metrics `AUC_Borji` 、`AUC_Judd` and `NSS`, due to the inconsistency caused by `resize/random/...` operations in `Python` and `Matlab`, we adopt the metrics calculated by ` Matlab`. **(Python ≠ Matlab)** **(We offer the official calculation method by Matlab.)**

​	👉 *If you wish to make predictions directly using our model results, you can do so using the proceeding steps.*

​	2. Test our model.

​		(a) You need to download our trained model file in table and put it to the specified folder path.   **(We are preparing. Please wait.)**

<div align="center">
<table>
  <thead>
    <tr>
      <th><i>$SalM^2$</i> for <i>TrafficGaze</i></th>
      <th><i>$SalM^2$</i> for <i>DrFixD-rainy</i></th>
      <th><i>$SalM^2$</i> for <i>BDDA</i></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="http://www.baidu.com" title="trained for TrafficGaze">trained for TrafficGaze</a></td>
      <td><a href="http://www.baidu.com" title="trained for DrFixD-rainy">trained for DrFixD-rainy</a></td>
      <td><a href="http://www.baidu.com" title="trained for BDDA">trained for BDDA</a></td>
    </tr>
  </tbody>
</table>
</div>

​		(b) You should use `--category` to switch datasets, which include `TrafficGaze`, `DrFixD-rainy`, `BDDA`. Run the following command.

```python
python evaluate_metrics.py --network salmm --b 1 --g 0 --category xxx --root xxx --test_weight xxx
```

​	👉If you are unable to adapt your environment for other reasons, you can also download our predictions directly.

​	3. Download prediction results.    **(We are preparing. Please wait.)**

<div align="center">
<table>
  <thead>
    <tr>
      <th><i>$SalM^2$</i> for <i>TrafficGaze</i></th>
      <th><i>$SalM^2$</i> for <i>DrFixD-rainy</i></th>
      <th><i>$SalM^2$</i> for <i>BDDA</i></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><a href="http://www.baidu.com" title="Prediction TrafficGaze">The prediction results link</a></td>
      <td><a href="http://www.baidu.com" title="Prediction DrFixD-rainy">The prediction results link</a></td>
      <td><a href="http://www.baidu.com" title="Prediction BDDA">The prediction results link</a></td>
    </tr>
  </tbody>
</table>
</div>

### 	(4) Run visualization

We also offer visualized code. Visualization can support the input of various types of data such as `str`, `list`, and `dataloader`.

​	1. input a dataloader

​	👉*If you want to visualize all the data of a certain dataset directly, you can use the following command.*

```python
python visualization.py --network salmm --b 1 --g 0 --category xxx --root xxx --test_weight xxx
```

​	2. input a string

​	👉*If you input a string (`str` : `'rootpath/0001/0001.jpg'`), please first modify the input in the visualization.py file, and then run the code.*

```python
python visualization.py --network salmm --b 1 --g 0 --category xxx --root xxx --test_weight xxx
```

​	3. input a list

​	👉*If you input a list (`list` : `['rootpath/0001/0001.jpg', 'rootpath/0001/0001.jpg', ......]`), please first modify the input in the visualization.py file, and then run the code.*

```python
python visualization.py --network salmm --b 1 --g 0 --category xxx --root xxx --test_weight xxx
```



If you find this repository useful, please use the following BibTeX entry for citation and give us a star⭐.

```python
@article{zhao2025salmamba, 
  title={SalM²: An Extremely Lightweight Saliency Mamba Model for Real-Time Cognitive Awareness of Driver Attention}, 
  volume={39}, 
  DOI={10.1609/aaai.v39i2.32157},  
  number={2},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
  author={Zhao, Chunyu and Mu, Wentao and Zhou, Xian and Liu, Wenbo and Yan, Fei and Deng, Tao}, 
  year={2025}, 
  month={Apr.}, 
  pages={1647-1655} 
}
```