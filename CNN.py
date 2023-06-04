import time
import numpy as np

import torch
import torch.nn as nn
import torchvision

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class myCNN(nn.Module):

    def __init__(self):
        super(myCNN, self).__init__()
        # grayscale image -> 1 input channel
        self.conv2d_0 = nn.Conv2d(in_channels=1, out_channels=6, stride=1, kernel_size=(3, 3))
        self.batch_0 = nn.BatchNorm2d(6)
        self.avg_0 = nn.AvgPool2d(kernel_size=(2, 2), stride=2)

        self.conv2d_1 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(3, 3), stride=1)
        self.batch_1 = nn.BatchNorm2d(16)
        self.avg_1 = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        # Edw flatten tin eksodo tou conv layer
        self.f0 = nn.Linear(in_features=24 * 24 * 16, out_features=120)
        self.f1 = nn.Linear(in_features=120, out_features=84)
        self.f2 = nn.Linear(in_features=84, out_features=10)

        self.ReLU = nn.ReLU()
        self.SoftMax = nn.Softmax()

    def forward(self, x):
        x = self.conv2d_0(x)
        x = self.batch_0(x)
        x = self.ReLU(x)

        x = self.conv2d_1(x)
        x = self.batch_1(x)
        x = self.ReLU(x)
        # Flatten
        x = x.view(x.size(0), -1)

        x = self.ReLU(self.f0(x))
        x = self.ReLU(self.f1(x))
        # The output of the f2 layer is
        x = self.SoftMax(self.f2(x))
        return x


def load_MNIST(b=32):
    tensor_transform = torchvision.transforms.ToTensor()

    train_data = torchvision.datasets.MNIST(root='./MNIST_dataset', transform=tensor_transform, train=True,
                                            download=True)
    test_data = torchvision.datasets.MNIST(root='./MNIST_dataset', transform=tensor_transform, train=False,
                                           download=True)

    train_load = torch.utils.data.DataLoader(train_data, batch_size=b, shuffle=True)
    test_load = torch.utils.data.DataLoader(test_data, batch_size=b, shuffle=False)

    return train_load, test_load


def train_and_eval_loop(train_data, test_data, model, loss_func, optimizer, epochs=100):
    print("Begin training on " + str(device) + ". . .")
    test_loss_per_epoch = []
    test_accuracy_per_epoch = []
    confusion_matrix = np.zeros((10, 10), dtype=int)
    for epoch in range(epochs):
        # Enable testing mode (has to do with Batch Normalization layers)
        start = time.time()
        model.train()
        epoch_training_loss = 0
        epoch_test_loss = 0
        epoch_test_accuracy = 0
        # Go through training data in batches with DataLoader
        for inputs, labels in train_data:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Clear the gradients in the optimizer
            optimizer.zero_grad()
            # Compute output labels
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
            # Calculate gradients with backward propagation
            loss.backward()
            # Update weights
            optimizer.step()
            # Sum the batch loss
            epoch_training_loss += loss.item()

        # Divide with the size of the training data and print the TRAINING epoch loss
        epoch_training_loss /= len(train_data)
        print(f'Epoch {epoch + 1}: training CE loss = {epoch_training_loss} | {time.time() - start:.1f}')

        # Enable evaluation mode (BatchNorm)
        model.eval()
        # Without calculating gradients
        with torch.no_grad():
            correct_class = 0
            total = 0
            for images, labels in test_data:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = loss_func(outputs, labels)
                epoch_test_loss += loss.item()
                # outputs Tensor contains all the possible classes in one dimension and the score for each class by the model
                # So to get the labels that the model assigned to the images, we need to find the max values of the tensor
                # Along the second dimension and get their indices (we do not need the actual values)
                max_values, max_indices = torch.max(outputs.data, dim=1)
                # Add to the cumulative sums of the confusion matrix
                for i in range(len(labels)):
                    true_label = labels[i].item()
                    predicted_label = max_indices[i].item()
                    confusion_matrix[true_label][predicted_label] += 1
                # Number of labels in current batch
                total += labels.size(0)
                # Make a new Tensor that contains True wherever the labels are the same with the indices
                hits = torch.Tensor(max_indices == labels)

                correct_class += torch.sum(hits).item()

            epoch_test_loss /= len(test_data)
            test_loss_per_epoch.append(epoch_test_loss)

            epoch_accuracy = 100.0 * correct_class / total
            test_accuracy_per_epoch.append(epoch_accuracy)

    print("Training Finished")
    return test_loss_per_epoch, test_accuracy_per_epoch, confusion_matrix


def compute_confusion_matrix(model, test_data, display_cond=False):
    print("Constructing Confusion Matrix. . .")
    # Initialize confusion matrix
    confusion_matrix = np.zeros((10, 10), dtype=int)
    # Model evaluation mode
    model.eval()
    with torch.no_grad():
        for images, labels in test_data:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            # outputs Tensor contains all the possible classes in one dimension and the score for each class by the model
            # So to get the labels that the model assigned to the images, we need to find the max values of the tensor
            # Along the second dimension and get their indices (we do not need the actual values)
            max_values, max_indices = torch.max(outputs.data, dim=1)
            # Add to the cumulative sums of the confusion matrix
            for i in range(len(labels)):
                true_label = labels[i].item()
                predicted_label = max_indices[i].item()
                confusion_matrix[true_label][predicted_label] += 1

    if display_cond:
        # Compute the row sums
        plt.figure()
        plt.imshow(confusion_matrix, cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()

        tick_marks = np.arange(confusion_matrix.shape[0])
        plt.xticks(ticks=tick_marks, labels=tick_marks)
        plt.yticks(ticks=tick_marks, labels=tick_marks)

        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                plt.text(j, i, str(confusion_matrix[i, j]),
                         ha="center", va="center",
                         color="white" if confusion_matrix[i, j] > confusion_matrix.max() / 2 else "black")
        plt.savefig("Confusion_Matrix_Color.png")
        plt.show()

    print("Confusion Matrix made. . .")
    return confusion_matrix


if __name__ == '__main__':
    # Load MNIST and split to batches
    train_loader, test_loader = load_MNIST(b=32)

    # Create model instance , optimizer and loss function
    model = myCNN()
    # Pass to GPU or CPU (whichever is available)
    model.to(device)
    # Cost function for classification problem and Stochastic Gradient Descent optimization
    loss = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(params=model.parameters(), lr=0.01)

    num_of_epochs = 50
    loss, accuracy, confusion_matrix = train_and_eval_loop(train_loader, test_loader, model, loss, opt,
                                                           epochs=num_of_epochs)
    # Save model after training
    torch.save(model, "myMNIST_CNN.pth")
    epochs = [i + 1 for i in range(num_of_epochs)]

    f, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(epochs, loss)
    ax1.set_title('Cross Entropy Loss on test set per epoch')

    ax2.plot(epochs, accuracy)
    ax2.set_title('Accuracy on test set per epoch')
    plt.tight_layout()
    plt.savefig("CNN_LOSS+ACCURACY")
    plt.show()

    m = torch.load('myMNIST_CNN.pth')

    cm = compute_confusion_matrix(m, test_loader, display_cond=True)
