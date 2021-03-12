# CNN-Aided Bit-Flipping for Belief Propagation Polar Decoder

We propose a convolutional neural network-aided bit-flipping (CNN-BF) mechanism to further enhance BP decoding of polar codes. It can achieve much higher prediction accuracy and better error correction capability than prior work of critical-set bit-flipping (CS-BF) but with only half latency. Hope this code is useful for peer researchers. If you use this code or parts of it in your research, please kindly cite our paper:

- **Related publication 1:** Chieh-Fang Teng, Andrew Kuan-Shiuan Ho, Chen-Hsi (Derek) Wu, Sin-Sheng Wong, and An-Yeu (Andy) Wu, "[Convolutional Neural Network-aided Bit-flipping for Belief Propagation Decoding of Polar Codes](https://arxiv.org/abs/1911.01704)," *published in 2021 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP).*

- **Related publication 2:** Chieh-Fang Teng and An-Yeu (Andy) Wu, "[Convolutional Neural Network-Aided Tree-Based Bit-Flipping Framework for Polar Decoder Using Imitation Learning](https://ieeexplore.ieee.org/document/9272860)," *published in 2021 IEEE Transactions on Signal Processing (TSP).*

---

## Required Packages

- python 3.6.5
- numpy 1.16.4
- tensorflow 1.8.0
- keras 2.2.5

## Source Code
- config.py: adjust parameters
  - `N` : Block length 
  - `K` : Information length
  - `in_N` : CRC's block length 
  - `in_K` : CRC's information length
  - `ebn0` : Desired SNR range 
  - `numOfWord` : Desired batch size 

- Generate Training Data.ipynb: generate the training data for CNN
  - `data_num` : Desired number of training data for each SNR 

- NN_BF.ipynb: use the generated training data to train CNN and show the prediction accuracy of both CNN-BF and CS-BF

- CS_BF.ipynb: repeat the simulatin results of critical-set belief propagation, need to set some parameters as below
  - `omega` : Number of flipped bit
  - `all_combination` : 0 for flipping and 1 for both flipping and strengthening
  - `T_max` : Maximum number of flipping trial and is initially set as the size of CS

- Analyze_Bound (progressive multi bit).ipynb: analyze the error correction capability of exhaustive BF (flipping and strengthing) as shown in Fig. 3 in [2](https://ieeexplore.ieee.org/document/9272860). Note that it is very slow...

- Analyze (progressive multi bit).ipynb: analyze the error correction capability of opposite BF (only flipping) as shown in Fig. 3 in [2](https://ieeexplore.ieee.org/document/9272860)

## Contact Information

   ```
Chieh-Fang Teng:
        + jeff@access.ee.ntu.edu.tw
        + d06943020@ntu.edu.tw
   ```
