# fault-detection
1.Dataset is downloaded from University of New South Wales which is showed in paper 'Rolling element bearing diagnostics using the Case Western
Reserve University data: A benchmark study'.

2.The structure of project:
(1)3metadata.txt and download.py are used to download data from CWRU.
(2)dataset.py reads the data.mat and then split the data into traindataset and validdataset.
(3)model.py consists of three kinds of deep neural network model for classfier and estimation of fault.
(4)fault diagnosis.py is the main function to call dataset and model, which train and test model for fault diagnosis.
(5)fault piagnosis.py is the main function to call dataset and model, which train and test model for fault piagnosis with trainmodel.py.
(6)fault diagnosis and estimate.py  is the main function for deployment, that load trained model and predict with current data.
(7)1D extract features.py and 2D extract features.py are used to visualize the extracted feature by neural network for explainable machine learning.

3.environment:
urllib;
numpy;
matplotlib;
scipy;
pytorch.
