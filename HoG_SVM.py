import time
import os
import joblib

import matplotlib.pyplot as plt
import numpy as np

from sklearn import svm
from sklearn.metrics import classification_report
from skimage.feature import hog
from skimage import filters
# Use torchvision to install (load) the data
from torchvision import datasets, transforms


def load_MNIST():
    tensor_transform = transforms.ToTensor()

    train_data = datasets.MNIST(root='./MNIST_dataset', transform=tensor_transform, train=True,
                                download=True)
    test_data = datasets.MNIST(root='./MNIST_dataset', transform=tensor_transform, train=False,
                               download=True)

    train_images = train_data.data.numpy()
    train_labels = train_data.targets.numpy()

    test_images = test_data.data.numpy()
    test_labels = test_data.targets.numpy()

    return train_images, train_labels, test_images, test_labels


def pad_image(image, pixels_per_cell):
    # scikit-image HOG does not pad the image by default
    image_height = image.shape[0]
    image_width = image.shape[1]

    patch_height, patch_width = pixels_per_cell
    if image_height % patch_height == 0 and image_width % patch_width == 0:
        return image
    # Calculate the right dimensions for an exact fit of patch size to image dimensions
    padded_height = int(np.ceil(image_height / patch_height) * patch_height)
    padded_width = int(np.ceil(image_width / patch_width) * patch_width)

    # Pad the image to match the right dimensions
    pad_height = padded_height - image_height
    pad_width = padded_width - image_width
    padded_image = np.pad(image, ((0, pad_height), (0, pad_width)), mode='edge')
    return padded_image.astype(dtype=np.float64)


def divide_matrix_to_blocks(image, block_size=(8, 8), stride=None):
    # If the stride is not specified, then the blocks are not overlapping. Assu
    if stride is None:
        stride = block_size[0]
    # Pad matrix (image) with zeros if block size does not perfectly fit the dimensions
    img = pad_image(image, block_size)
    num_blocks_height = (img.shape[0] - block_size[0]) // stride + 1
    num_blocks_width = (img.shape[1] - block_size[1]) // stride + 1
    blocks = np.zeros((num_blocks_height, num_blocks_width, block_size[0], block_size[1]))

    for i in range(num_blocks_height):
        for j in range(num_blocks_width):
            y_start = i * stride
            y_end = y_start + block_size[0]
            x_start = j * stride
            x_end = x_start + block_size[1]

            blocks[i, j] = image[y_start:y_end, x_start:x_end]
    # Reshape to 3 dimensions because we do not care about location of block in image
    blocks = blocks.reshape((-1, block_size[0], block_size[1]))
    return blocks


def get_gradients(image):
    grad_x = filters.sobel_h(image)
    grad_y = filters.sobel_v(image)

    # Calculate the magnitude and angle of the gradients
    grad_magnitude = np.sqrt(np.square(grad_x) + np.square(grad_y))
    # Angle from radians to degrees
    grad_angle = np.arctan2(grad_y, grad_x) * 180 / np.pi

    return grad_magnitude, grad_angle


def create_histogram(block_magnitudes, block_angles):
    hist = np.zeros(9)
    center_angles = [10, 30, 50, 70, 90, 110, 130, 150, 170]
    num_of_pixels = len(block_magnitudes) * len(block_magnitudes)
    pixel_values = np.stack((block_magnitudes, block_angles), axis=-1).reshape(num_of_pixels, 2)
    print("")


# Calculate HOG. Expects a numpy array of images
def calculate_HOG(images, patch_size=(8, 8), block_size=(2, 2)):
    print("Calculating HOG . . .")
    start = time.time()
    feature_vectors = []
    for i in range(images.shape[0]):
        # Obtain image
        img = images[i]
        # Pad the image to allow for all different patch sizes
        # img = pad_image(img, pixels_per_cell=patch_size)
        # Calculate HOG of Image
        hist = hog(image=img, pixels_per_cell=patch_size, cells_per_block=block_size)
        feature_vectors.append(hist)
    print(f'Finished calculations after {time.time() - start:.1f}s')
    return np.array(feature_vectors)


def compute_confusion_matrix(labels, predictions, save_folder):
    print("\tMaking Confusion Matrix")
    size = len(labels)
    matrix = np.zeros((10, 10), dtype=np.int32)
    for i in range(size):
        true_label = labels[i]
        pred_label = predictions[i]
        matrix[true_label, pred_label] += 1
    if save_folder:
        plt.figure()
        plt.imshow(matrix, cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        # Make the axis display all values
        tick_marks = np.arange(matrix.shape[0])
        plt.xticks(ticks=tick_marks, labels=tick_marks)
        plt.yticks(ticks=tick_marks, labels=tick_marks)
        # Write the number of pairs in each grid
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                plt.text(j, i, str(matrix[i, j]),
                         ha="center", va="center",
                         color="white" if matrix[i, j] > matrix.max() / 2 else "black")
        plt.savefig(os.path.join(save_folder, "Confusion_Matrix_Color.png"))
        plt.show()
    print("\t Confusion Matrix computed")
    return matrix


if __name__ == '__main__':

    train_images, train_labels, test_images, test_labels = load_MNIST()
    patch_sizes = [4, 8, 12, 14]

    for size in patch_sizes:
        folder_to_save = os.path.join(os.getcwd(), "SVM_patch" + str(size))
        train_file = os.path.join(folder_to_save, "train_vectors.npy")
        test_file = os.path.join(folder_to_save, "test_vectors.npy")
        model_file = os.path.join(folder_to_save, "model.pkl")
        report_file = os.path.join(folder_to_save, "evaluation_report.txt")
        if not os.path.exists(folder_to_save):
            os.makedirs(folder_to_save)
        if not os.path.exists(train_file):
            train_vectors = calculate_HOG(train_images, patch_size=(size, size))
            print(train_vectors.shape)
            test_vectors = calculate_HOG(test_images, patch_size=(size, size))
            np.save(train_file, train_vectors)
            np.save(test_file, test_vectors)
        else:
            print("HOG Feature Vectors loaded from model directory")
            train_vectors = np.load(file=train_file)
            test_vectors = np.load(file=test_file)
        # Initialize and train model
        if not os.path.exists(model_file):
            start = time.time()
            SVM_model = svm.SVC()
            print(f'Training SVM with patch {size}x{size} . . .')
            SVM_model.fit(train_vectors, train_labels)
            print(f'Training finished after {time.time() - start:.1f}s')
            joblib.dump(SVM_model, model_file)
        else:
            print("Model already trained and loaded")
            SVM_model = joblib.load(filename=model_file)
        # Get predictions and confusion matrix
        test_predictions = SVM_model.predict(test_vectors)
        report = classification_report(y_true=test_labels, y_pred=test_predictions, labels=[i for i in range(10)],
                                       digits=5)
        confusion_matrix = compute_confusion_matrix(test_labels, test_predictions, save_folder=folder_to_save)
        print(report)
        with open(report_file, "w") as f:
            f.write(report)
