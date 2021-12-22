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
* tqdm==4.62.3
* git+https://github.com/thu-ml/zhusuan.git
* git+https://github.com/haowen-xu/tfsnippet.git@v0.2.0-alpha1

## Datasets
### Link
The used datasets are available at:
* MSL&SMAP https://github.com/khundman/telemanom
* SWaT&WADI https://itrust.sutd.edu.sg/itrust-labsdatasets/datasetinfo/

### Preprocess the data
`python data_preprocess.py <dataset>`

where `<dataset>` is one of `SMAP`, `MSL`, `SWaT` and `WADI`, then you will get `<dataset>_train.pkl`, `<dataset>_test.pkl` and `<dataset>_test_label.pkl` in folder ‘processed’.

## Run Code
`python main.py`

If you want to change the default configuration, you can edit `ExpConfig` in `main.py`. For example, if you want to change dataset, you can change the value of 'dataset'.
