# DendroID
Digital identification of wood grain pattern using a Siamese CNN
Repository for the CAS_ML. Due to size restriction of this repository, only a part of the data are available. The complete dataset can be downloaded under Swiss Transfer Link (available for 30 days):
XX
Before starting the script, consider reading the requirements written below:
Python 3.12.3
TensorFlow: 2.18.0
NumPy: 2.0.2
Scikit-learn: 1.6.1
Matplotlib: 3.10.3

Please note that all calculations were made on a wsl ubuntu machine running on windows to allow for GPU calculation.
If needed, use the script "crop_and_orient_FINAL.py" on raw image to isolate the contour and correct the perspective.
of the cards. However, all images contained in the repository are already cropped and ready to use. 
The core of the work is contained in the "woodsiamese.ipynb" Jupyter notebook. If one just want to test the performance of the model, please skip directly to the optional section "Model Import", where a saved keras model can be imported and evaluated. 

No part of this project should be copied without prior approval of the author. 
Please contact me at clerc@swisswoodsolutions.ch for questions.
