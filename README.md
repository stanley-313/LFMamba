# LFMamab: Light Field Image Super-Resolution with State Space Model
***
![](/figs/network.png)
This is the Pytorch implementation of the LF image spatial SR method in 
our paper "Omni Spatial-Angular Correlations Exploration for Light Field Image Super-Resolution".

## Preparation:
***

1. Requirement:
   - pytorch = 1.13.0, torchvision = 0.13.0, python = 3.10, cuda = 11.7
   - causal-conv1d = 1.1.1
   - mamba-ssm = 1.1.2
2. Datasets:
   - We use five LF benchmarks in [BasicLFSR](https://github.com/ZhengyuLiang24/BasicLFSR)
   (i.e., EPFL, HCInew, HCIold, INRIA, and STFgantry). Download and put them in folder `./datasets/`.
3. Generate training and testing data:
   - Run `Generate_Data_for_Training.py` to generate training data in `./data_for_training/`.
   - Run `Generate_Data_for_Test.py` to generate testing data in `./data_for_test/`.
   
## Train:
***
- Set the hyper\-parameters in `parse_args()` in `train.py` if needed.
- Run `train.py` to train network
- Checkpoints will be saved in `./log/`

## Test:
***
- Run `test.py` to perform test on each dataset. The resultant `.mat` files will be saved in `./Results/`
- Run `GenerateResultImages.py` to generate SR RGB images. Saved in `./SRimage/` 
### Results:
***
#### Quantitative results:

![](/figs/quantitative.png)

#### Visual comparisons:
![](/figs/visual_results.png)

### Efficiency:
![](/figs/efficiency.png)

### Angular consistency:
![](/figs/angular_consistency.png)

### LFMamba for LF angular SR
**Quantitative results:**
![](/figs/ASR_quantitative.png)
**Visual comparisons:**
![](/figs/ASR_visual.png)

## Contact us:
*** 
For any questions, please email to [wangxia@bit.edu.cn](wangxia@bit.edu.cn).
