# MNIST-CNN-SVM-Classifiers-with-HoG-Feature-Extraction

A python project on the MNIST handwritten digits dataset (http://yann.lecun.com/exdb/mnist/). 

In the CNN.py script, a CNN is created and trained on MNIST using the Pytorch module. The model is then evaluated on the test dataset. Cross-Entropy Loss and accuracy are calculated, as well as a Confusion Matrix. The plots are saved in the project directory after execution. The architecture of the CNN is shown in the following image:

![image](https://github.com/petrakisgeo/MNIST-CNN-SVR-Classifiers-with-HoG-Feature-Extraction/assets/117226445/97ec68d9-8e37-4b9a-be8b-5d6085e90a01)

In the HoG_SVM.py, we extract feature vectors of the MNIST images from their Histogram of oriented Gradients, using the scikit-image module. Then an SVM is trained and evaluated. The testing-evaluation process is performed for different patch sizes for the calculation of the histograms. After execution of the script, the feature vectors, the models and the results, containing Confusion Matrices in .png format are saved in different directories depending on the patch size.

Example image of a confusion matrix from the results of an SVR evaluation:

![image](https://github.com/petrakisgeo/MNIST-CNN-SVR-Classifiers-with-HoG-Feature-Extraction/assets/117226445/0d3e5961-a4b9-4111-b499-e99a5a8b8e7a)


Also, a personal re-implementation of the scikit-image hog() function is presented in a different repository.

packages.txt for conda virtual environment build
requirements.txt for pip virtual environment build
