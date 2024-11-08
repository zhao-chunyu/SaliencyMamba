

# Supplementary materials



### (1) We release all the runnable code.

### (2) Comparison of Runtime and GPU memory.

We evaluated our method's runtime and memory usage across different batch sizes and compared it to the lightweight CDNN network for efficiency assessment. Results indicate that while both models' memory usage increases with batch size, our model grows more gradually. At batch size 256, our model uses under 10GB GPU memory, whereas CDNN surpasses 45GB. Runtime analysis further shows our model’s runtime remains stable around 100ms with increasing batch size, whereas CDNN’s runtime escalates sharply, reaching nearly four times our model's runtime at batch size 256.

<div align="center">
<img src="fig\runtime.jpg" width="900" height="auto" />
</div>



### (3) Driver attention shift cases. (+15 cases)

When such stimuli induce attention shifts, driver attention manifests as changes in the primary region of interest or the emergence of secondary and tertiary fixation points. While Figure 5 (rows 2, 3, and 6) illustrates attention shifts towards road signs, incoming vehicles, and traffic lights, it may not comprehensively reflect shifts in the primary fixation region. To address this gap, we intend to enhance the revised version with additional driver attention shift cases.

<div align="center">
	<img src="fig\case1.jpg" width="900" height="auto" />
    <img src="fig\case2.jpg" width="900" height="auto" />
    <img src="fig\case3.jpg" width="900" height="auto" />
</div>

### (4) Performance of different resolution for our model. (TrafficGaze)

As shown in the table below, the experimental results demonstrate that: these metrics result in only a slight improvement over 256×256 input images. However, it has a FLOPs of 4.71, which significantly increases the computational amount.

| Image size | AUC_B↑ | AUC_J↑ | NSS↑ | CC↑  | SIM↑ | KLD↓ | FLOPs↓ |
| ---------- | ------ | ------ | ---- | ---- | ---- | ---- | ------ |
| 3×256×256  | 0.92   | 0.98   | 5.90 | 0.94 | 0.78 | 0.28 | 4.45   |
| 3×512×512  | 0.92   | 0.98   | 6.04 | 0.95 | 0.80 | 0.26 | 4.71   |
