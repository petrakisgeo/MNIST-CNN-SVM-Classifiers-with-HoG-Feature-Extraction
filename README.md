# MNIST-CNN-SVR-Classifiers-with-HoG-Feature-Extraction

A python project on the MNIST handwritten digits dataset http://yann.lecun.com/exdb/mnist/. Torchvision module is used to install/load the dataset in both scripts

In the CNN.py script, a CNN is created and trained on MNIST using the Pytorch module. The model is then evaluated in the test dataset. Cross-Entropy Loss and accuracy are calculated, as well as a Confusion Matrix. The plots are saved in the project directory after execution

In the HoG_SVM.py, we extract feature vectors of the MNIST images from their Histogram of oriented Gradients, using the scikit-image module. Then an SVR model is trained and evaluated. The testing-evaluation process is performed for different patch sizes for the calculation of the histograms. After execution of the script, the feature vectors, the models and the results, containing Confusion Matrices in .png format are saved in different directories depending on the patch size.

A different implementation of the scikit-image hog() function is presented in a different repository, trying to 
