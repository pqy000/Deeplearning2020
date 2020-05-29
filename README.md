# Deep Learning 2020
Course project of Deep learning 2020 Spring

## Time series forcasting for car dataset

# Dataset

Dataset can be downloaded from 

```https://cloud.tsinghua.edu.cn/d/8d99739d05f0464f8986/  ```

If there are something wrong with the Data, please go ahead

``` https://github.com/laiguokun/multivariate-time-series-data/  ``` 


The car dataset is from (547M)

```https://cloud.tsinghua.edu.cn/f/1c1e8349a7a747fea07a/?dl=1```

### Environment 

* Python 3.6+
* Pytorch 1.0+
* numpy

### Example

1. `Traffic dataset: ` traffic.sh
2. `Solar-Energy dataset:`solar.sh
3. ` Electricity usage dataset:` ele.sh 

### Instruction

main.py  
* --data DATA                     `location of the data file `
* -h --help                       ` show this help message and exit `
* --model DATA                    ` select the model： LSTNet, CNN, RNN or MHA_Net `
* --window WINDOW                 ` window size (history size) `
* --horizon HORIZON               `forecasting horizon(step) `
* --hidRNN HIDRNN                 `number of RNN hidden units each layer`
* --rnn_layers RNN_LAYERS         ` number of RNN hidden layers`
* --hidCNN HIDCNN                 ` number of CNN hidden units (channels)`
* --CNN_kernel CNN_KERNEL         `the kernel size of the CNN layers`
* --highway_window HIGHWAY_WINDOW `The window size of the highway component `  
* -n_head N_HEAD                  `num of self attention heads                     `
* -d_k D_K                        `self attention key dimension`
* -d_v D_V                        `self attention value dimension`
* --clip CLIP                     `gradient clipping limit `
* --epochs EPOCHS                 `upper epoch limit `
* --batch_size N                  `batch_size`
* --dropout DROPOUT               `dropout applied to layers (0 = no dropout)`
* --seed SEED                     `random seed`
* --log_interval N                `report interval`
* --save SAVE                     `path to save the final model'`
* --cuda CUDA                     `whether to use cuda device`
* --optim OPTIM                   `optimizer method ,default 'adam'`
* --amsgrad AMSGRAD               `whether to use amsgrad`
* --lr LR                         `learning rate`
* --skip SKIP                     `autoregression window size`
* --hidSkip HIDSKIP               `skiphidden states dimension`
* --L1Loss L1LOSS                 `whether  to use l1 loss function`
* --normalize NORMALIZE           `whether  to normalize the data`
* --output_fun OUTPUT_FUN         `relu, tanh or sigmoid `




