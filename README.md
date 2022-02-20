<p align="center">
<img src="https://raw.githubusercontent.com/norse/norse/master/logo.png" alt="Norse logo"/>
</p>     

# Norse Notebook Tutorials

These notebooks are designed to familiarize yourself with the [Spiking Neural Network simulator, Norse](https://github.com/norse/norse).
This material can be **run the notebooks directly in your browser**.
However, please note that the online execution relies on CPU, which can cause performance problems for larger networks.
If you desire hardware acceleration (like GPU or TPU) you can either clone this repository and experiment locally, or use Google Colab.

If this is your first brush with computational neuroscience, we recommend [this excellent serious of tutorials by Neuromatch Academy](https://compneuro.neuromatch.io/tutorials/W0D1_PythonWorkshop1/student/W0D1_Tutorial1.html).
They provide excellent interactive tutorials on neuron dynamics, linear algebra, calculus, statistics, and deep learning.

## Level: Beginner

| Notebook | Topic | 
| ------------------ | --- | 
| [Introduction to PyTorch and spiking neurons](https://norse.github.io/notebooks/intro_spikes.html) | Introduces biological neurons and PyTorch | 
| [Introduction to Spiking Neural Networks in Norse](https://norse.github.io/notebooks/intro_norse.html) | Build and train spiking models in Norse |
| [Simulating and plotting spiking data](https://norse.github.io/notebooks/intro_plotting.html) | Learn how to describe and visualise event data |
| *Encoding data to spikes* | Learn how to encode conventional data into spikes |
| *Single Neuron Experiments* | Learn how single neurons behave |

## Level: Intermediate

| | | |
| ------------------ | --- | --- |
| *Training on MNIST* | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/norse/notebooks/blob/master/mnist_classifiers.ipynb) | Learn how to solve MNIST with spikes
| *Working with event-based data* | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/norse/notebooks/blob/master/poker-dvs_classifier.ipynb) | Learn how to work with event-based datasets by classifying a set of poker cards
| *Optimization of neuron parameters* | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/norse/notebooks/blob/master/parameter-learning.ipynb) | Explore how automatic differentiation can be useful to optimize neurons


## Level: Advanced

| | | |
| ------------------ | --- | --- |
| *High Performance Computing with Norse* | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/norse/notebooks/blob/master/high-performance-computing.ipynb) | Scale Norse models to HPCs! |
| *Using plasticity in Norse* | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/norse/notebooks/blob/master/stp_example.ipynb) | Learn how plasticity works in Norse |
| *Stochastic Computing* | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/norse/notebooks/blob/master/stochastic-computing.ipynb) | Explore stochastic computing with spiking neurons |

For more information we refer to our [documentation](https://norse.ai/docs) as well as [our suite of tasks](https://github.com/norse/norse/task/).

## Flowchart

Don't know where to start? Follow this flow chart to get you going!
**Please note** that we assume basic familiarity with Python.

```mermaid
flowchart TD
    Background[What are you most familiar with?]
    Background -- Computer Science --> Spikes
    Background -- Machine learning --> Encoding
    Background -- Neuroscience --> PyTorch

    Spikes[]
    Encoding
    PyTorch[<a href='https://colab.research.google.com/github/norse/notebooks/blob/master/intro_norse.ipynb'>Introduction to spiking networks with Norse <img alt='Open in Google Colab' src='https://colab.research.google.com/assets/colab-badge.svg' style='width:100px'/></a>]

    More[What would you like to know more about?]
    Spikes --> More
    Encoding --> More
    PyTorch --> More

    SL{Supervised learning}
    DVS{Event-based vision}
    Neuroscience{Neuroscience}

    More --> SL 
    SL --> MNIST
    SL --> HPC
    SL --> Poker

    More --> DVS
    DVS --> Poker
    DVS --> Aestream
    
    More --> Neuroscience
    Neuroscience --> Parameters
    Neuroscience --> Plasticity
    Neuroscience --> Visualization
```

We encourde you to explore the [main repository](https://github.com/norse/norse/) and contribute by either improving the tutorials, write code, or donate to support our work. Thank you!







