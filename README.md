# MNIST-From-Scratch
Created a neural network using numpy to detect numbers from the mnist database (no ML libraries, as I wanted to learn how neural networks work)

## Structure
Consists of Input and Output Layer as well as 2 Hidden Layers  
Tried implementing gradient descent with momentum  
Input Layer : 784 neurons  
Hidden Layer 1 : 16 neurons, Sigmoid activation function  
Hidden Layer 2 : 16 neurons, ReLU activation function  
Output Layer : 10 neurons, softmax activation function  

## Results
Saved the weights and biases of the trained model in [parameters.npz](https://github.com/202248SD/MNIST-From-Scratch/blob/c20a58fcecb93700bb3ac3cdaf326b8bfe80dace/parameters.npz)
Train Accuracy: ~87.2%, Cost: ~2.2  
Test Accuracy: ~79.3%, Cost: ~3.5  
Test Accuracy 8% Lower than train accuracy, which might be due to overfitting  

## Resources
For this project, I used some youtube videos as well as stackoverflow whenever I was stuck with the maths  
### [3B1B](https://www.youtube.com/@3blue1brown)  
[But what is a neural network? | Chapter 1, Deep learning](https://youtu.be/aircAruvnKk?si=jXTfAIkC7rJNzprT)  
[Gradient descent, how neural networks learn | Chapter 2, Deep learning](https://youtu.be/IHZwWFHWa-w?si=Hq_DYZU6GB1QBZi8)  
[What is backpropagation really doing? | Chapter 3, Deep learning](https://youtu.be/Ilg3gGewQ5U?si=uQNzuVudPVPZfjSH)  
[Backpropagation calculus | Chapter 4, Deep learning](https://youtu.be/tIeHLnjs5U8?si=j_i3pSUsKNP0NQ1D)
### [Samson Zhang](https://www.youtube.com/@SamsonZhangTheSalmon)  
[Building a neural network FROM SCRATCH (no Tensorflow/Pytorch, just numpy & math)](https://youtu.be/w8yWXqWQYmU?si=tOt2nHiJfen5tr3z)  
