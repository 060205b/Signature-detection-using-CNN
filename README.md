# Signature-detection-using-CNN

**Signature Classification Using VGG16**
This project implements a signature classification model using the VGG16 architecture, pre-trained on ImageNet. The model is fine-tuned to classify images of signatures into different classes. The project uses PyTorch and torchvision libraries for loading the dataset, transforming the images, and training the model.

Required tools 
Dataset Preparation
Model Training
Image Classification
Results
License

Required tools 
To run this project, you need to have Python installed along with the following libraries:

```pip install torch torchvision pillow```

Requirements
Python 3.x
PyTorch
torchvision
PIL (Python Imaging Library, included in the pillow package)

Dataset Preparation
Dataset Structure:

The dataset should be organized in the following structure:
Each subfolder in the sign directory represents a class (e.g., different people’s signatures).

Set the Training Folder:
Update the train_folder variable in the script to point to your dataset directory.

```train_folder = "C:/path/to/your/dataset/sign"```

Model Training
Pre-trained Model:
The script uses the pre-trained VGG16 model provided by torchvision. The last fully connected layer is replaced to match the number of classes in your dataset.

Training Configuration:
Batch Size: Set to 3.
Number of Epochs: Set to 9.
Optimizer: Adam optimizer with a learning rate of 0.001.
Learning Rate Scheduler: Reduces the learning rate by a factor of 0.1 every 7 epochs.

Training Process:
The model is trained on the specified dataset with the given configurations.
Features of the VGG16 model are frozen to leverage the pre-trained weights.

    for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    exp_lr_scheduler.step()
   
Image Classification
After training the model, you can classify a new image using the classify_image function.

Image Classification:
Provide the path to the image you want to classify.
The image is processed and passed through the model to predict the class.
The predicted class and its probability are printed.

Results
The model’s performance can be evaluated by the accuracy of the predictions on test images.
During training, the learning rate is adjusted using a scheduler to improve model performance.


```
img_path = r"C:\path\to\your\image.png"
classify_image(img_path)```

```Example Output:
Prediction: Mithul with probability: 85.23%```



