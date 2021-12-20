# MUTANT
For TNNLS‘22-submission
> Robust Anomaly Detection for Multivariate Time Series through Temporal GCNs and Attention-based VAE

## Dependencies
Recent versions of the following packages for Python 3 are required:
* numpy==1.21.2
* torch==1.9.1
* scipy==1.7.1
* scikit-learn==0.24.2
* pandas==0.25.0

## Datasets
### Link
The used datasets are available at:
* MSL&SMAP https://github.com/khundman/telemanom
* SWaT&WADI https://itrust.sutd.edu.sg/itrust-labsdatasets/datasetinfo/

### Preprocess the data
python data_preprocess.py <dataset> 

where <dataset> is one of SMAP, MSL, SWaT, WADI


In addition, we also sample the positive and negative edges in the network, and divide them into three text files: train, valid and test for link prediction.

## Usage
First, you need to determine the data set. If you want to do node classification tasks, you need to modify the data set path in `Node_classification.py`. If you want to do link prediction, you need to modify the dataset path in `Link_prediction.py`.
