# Official calculation method

***[note] First of all, you need to install MATLAB on your PC. Please find some tutorials by yourself.***

### (1) Save prediction file to `.mat`

```python
python saveMat.py --network salmm --b 1 --g 0 --category xxx --root xxx --test_weight xxx
```

### (2) Calculate metrics by `Matlab`

You can calculate the indicators by using either **(a) command line** or **(b) GUI**.

[note] If you are running the code on a remote server (ssh), it is recommended to use the command line.
[note] If you have access to the server with a UI interface, it is recommended to use the GUI.

### 	(a) Run by the command line

*[note] We are running on the Ubuntu server.*

* Add executable permissions for the `run_matlab.sh` file

```python
cd metrics
chmod +x run_matlab.sh
```

* Change the location of the MATLAB executable file in `run_matlab.sh` file.
	We offer a reference :

```python
MATLAB_BIN="/home/zcy/matlab/bin/matlab"
```
* Modify the parameters in the params.txt file.

    - **1st line** : Dataset categories (`BDDA`, `TrafficGaze`, `DrFixD_rainy`)
    - **2nd line**: Root directory of the output of the `.mat` file
    - **3rd line** : fixdata (`TrafficGaze`, `DrFixD_rainy`) or the root directory of the output of the `.mat` file (`BDDA`)
    - **4th line** : Name file of all test sets (converted from `test.json`)

    We offer a reference for `BDDA`:

```python
BDDA
ckpts/BDDA/salmm_20240704-15:46:29/outputMat/
ckpts/BDDA/salmm_20240704-15:46:29/outputMat/
dataset/BDDA/test.mat
```

​		We offer a reference for `TrafficGaze`:

```python
TrafficGaze
ckpts/TrafficGaze/salmm_20240704-15:46:29/outputMat/
dataset/TrafficGaze/fixdata/fixdata
dataset/TrafficGaze/fixdata/test.mat
```

​		We offer a reference for `DrFixD_rainy`:

```python
DrFixD_rainy
ckpts/DrFixD_rainy/salmm_20240704-15:46:29/outputMat/
dataset/DrFixD_rainy/fixdata/fixdata
dataset/DrFixD_rainy/fixdata/test.mat
```

* Run the `run_matlab.sh` file

```python
./run_matlab.sh
```

### 	(b) Run by the Matlab GUI

*[note] Using the Maltab GUI for running is convenient, but it is not suitable for use in ssh connection.*

* Initialize 4 parameters in the `runForGUI.m`

```python
data_cls=''
path1=''
path2=''
path3=''
```

* Run the script `runForGUI.m` in Matlab






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

