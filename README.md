# AdvancedAI
# Image Classification using Convolutional Neural Networks (CNN) and Fully Connected Neural Networks (FCNN)

This project comprehensively implements image classification using Convolutional Neural Networks (CNN) and Fully Connected Neural Networks (FCNN). This project aims to accurately classify fashion items from the Fashion-MNIST dataset into ten different classes. The code covers various stages of the image classification pipeline, including data preprocessing, model architecture creation, model training, parameter tuning, and model evaluation.


## Dataset Exploration

The Fashion-MNIST dataset consists of 60,000 training and 10,000 testing samples, each representing grayscale images of fashion items like t-shirts, trousers, pullovers, dresses, coats, sandals, shirts, sneakers, bags, and ankle boots. Before diving into model development, the dataset distribution is visualised, and the reader is presented with sample images from each class.
<img width="1004" alt="Code used to display the spread of items within the dataset" src="https://github.com/Quints497/AdvancedAI/assets/70848538/948bed3d-5e72-49c8-b8bf-1db1704b47ac">
<img width="1004" alt="Code used to display a sample image from each class" src="https://github.com/Quints497/AdvancedAI/assets/70848538/f9fbaea7-6fdc-4b06-ba50-5dae1e308c8d">


## Data Preprocessing

To ensure optimal model performance, data preprocessing is crucial. The pixel values of the images, originally ranging from 0 to 255, are scaled to the 0 to 1 range, making the inputs compatible with neural networks. Additionally, the images are reshaped to have a depth of 1 to form 1-dimensional arrays for network input. the labels are transformed into binary vectors using one-hot encoding, facilitating categorical classification.
<img width="1004" alt="Code used to display how the data has been processed" src="https://github.com/Quints497/AdvancedAI/assets/70848538/c7d35b3a-1494-435f-b014-65a3fa12d313">


## CNN Architecture Exploration

The report explores various CNN architectures with different numbers of convolutional and pooling layers. Each layer uses ReLU activation, while the output layer employs Softmax activation for classification. To prevent overfitting, early stopping is implemented during training. The CNN models' performance is assessed based on Categorical Croseentropy loss, Categorical Accuracy, and Precision metrics.
<img width="1004" alt="Code used to create various CNN architectures" src="https://github.com/Quints497/AdvancedAI/assets/70848538/a5c159d7-9af5-4980-bd3a-105007340804">


## FCNN Architecture Exploration

Similarly, FCNN models are created with differing numbers of fully connected layers. To counter overfitting, Dropout layers are introduced, which randomly deactivate neurons during training. This allows the model to generalise better to unseen data.
<img width="1004" alt="Code used to create various FCNN architectures" src="https://github.com/Quints497/AdvancedAI/assets/70848538/57269bcd-c17a-402a-9a53-8b4f39197da0">


## Model Training and Performance

The models are trained on the training data with a validation split to monitor performance during training. Categorical Croseentroppy loss is used to optimise the models, while Categorical Accuracy and Precision are monitored to assess their accuracy and precision on the validation set. Early stopping is incorporated to prevent overfitting and achieve the best model performance.
<img width="1004" alt="Code used to create the Training Function used by the CNN and FCNN models" src="https://github.com/Quints497/AdvancedAI/assets/70848538/2102fb8d-e8ec-4f8e-9645-3a0329f6fa71">
<img width="1004" alt="Code used to create the Visualisation Function used by the CNN and FCNN models" src="https://github.com/Quints497/AdvancedAI/assets/70848538/4c5302bb-a6cd-4dfe-be73-297cbe045c8d">


## Hyperparameter Tuning

To find the optimal hyperparameters for both CNN and FCNN models, a parameter tuning script is employed. Parameters such as the number of filters/units, dropout rate, and learning rate are systematically explored using a grid search approach. This step ensures the models are fine-tuned for maximum accuracy.


## Model Evaluation

After training with the optimal hyperparameters, the models' performance is evaluated on the testing dataset to assess their generalisation capabilities. the models' loss, accuracy, and precision are measured on the unseen data.
<img width="1004" alt="Code used to test and then display the model's performance" src="https://github.com/Quints497/AdvancedAI/assets/70848538/bb107c20-a8e3-4f12-b7e7-d42a05d7b17f">
<img width="1004" alt="The loss, accuracy, precision score of CNN & FCNN models" src="https://github.com/Quints497/AdvancedAI/assets/70848538/ffeeb947-3009-400b-ab63-0328df494098">


## Comparison with Literature

To gain perspective on the model's performance, the final CNN and FCNN model's test accuracy is compared with other models from literature. This comparative analysis highlights the CNN and FCNN model's competitive results compared to state-of-the-art models but also acknowledges the increased computational costs associated with CNNS.

<img width="1004" alt="Displays the comparison of scores between the CNN, FCNN, and models found in literature" src="https://github.com/Quints497/AdvancedAI/assets/70848538/b53e0e8c-4ab4-4967-97ec-d69128ba7e70">


## Applications of CNNs

The report contains insights into real-world applications of CNNs, showcasing their versatility in tasks such as object detection, semantic segmentation, and image captioning. CNNs have become a standard in computer vision applications and are widely used in cutting-edge technology.


## Conclusion

The CNN model demonstrates superior performance in terms of accuracy and loss compared to the FCNN model. Regularisation techniques, including dropout layers, are implemented to mitigate overfitting and improve generalisation. The CNN model's competitive results, as compared to models from literature, demonstrate its effectiveness in fashion item classification. However, it is essential to consider the computational resources required for CNN training, as they can be more resource intensive compared to FCNNs



