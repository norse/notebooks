<p align="center">
<a href="https://norse.github.io/norse"><img src="https://raw.githubusercontent.com/norse/norse/master/logo.png" alt="Norse logo"/></a>
</p>

# Norse Notebook Tutorials

These notebooks are designed to familiarize yourself with the [Spiking Neural Network simulator, Norse](https://norse.github.io/norse/).
These notebooks can be [directly run in your browser](https://norse.github.io/notebooks/README.html#).
However, please note that the online execution relies on CPU, which can cause performance problems for larger networks.
If you desire hardware acceleration (like GPU or TPU) you can either clone this repository and experiment locally, or use Google Colab.

If this is your first brush with computational neuroscience, we recommend [this excellent serious of tutorials by Neuromatch Academy](https://compneuro.neuromatch.io/tutorials/W0D1_PythonWorkshop1/student/W0D1_Tutorial1.html).
They provide interactive tutorials on neuron dynamics, linear algebra, calculus, statistics, and deep learning.

## Introductory notebooks

Create

| Notebook | Topic |
| ------------------ | --- |
| [Introduction to PyTorch and spiking neurons](https://norse.github.io/notebooks/intro_spikes.html) | Introduces biological neurons and PyTorch |
| [Introduction to Spiking Neural Networks in Norse](https://norse.github.io/notebooks/intro_norse.html) | Build and train spiking models in Norse |
| [Simulating and plotting spiking data](https://norse.github.io/notebooks/intro_plotting.html) | Learn how to describe and visualise event data |


## Supervised Learning

| Notebook | Topic |
| ------------------ | --- |
| [Training with MNIST](https://norse.github.io/notebooks/mnist_classifiers.html) | Learn how to solve MNIST with spikes
| [Learning event-based DVS Poker](https://norse.github.io/notebooks/poker-dvs_classifier.html) | Learn how to work with event-based datasets by classifying a set of poker cards


## Real-time event processing

| Notebook | Topic |
| ------------------ | --- |
| [Edge detection with Norse](https://norse.github.io/notebooks/edge_detector.html) | Process events from `.aedat4` files with Norse. Prerequisite to streaming real-time events. |

## Neuroscience
| Notebook | Topic |
| ------------------ | --- |
| [Optimizing neuron parameters](https://norse.github.io/notebooks/parameter-learning.html) | Learn how to solve MNIST with spikes
| [Spike time dependent plasticity](https://norse.github.io/notebooks/stp_example.html) | Learn how to work with event-based datasets by classifying a set of poker cards


## Miscellaneous
| Notebook | Topic |
| ------------------ | --- |
| [High Performance Computing with Norse](https://norse.github.io/notebooks/high-performance-computing.html)  | Scale Norse models to HPCs! |
| [Stochastic Computing](https://norse.github.io/notebooks/stochastic-computing.html) | Explore stochastic computing with spiking neurons |

For more information we refer to our [documentation](https://norse.ai/docs).

We are also more than happy to accept contributions such as improving or adding notebooks, suggestions for improvements, issues for bugs, or donations to support our work. Thank you!

## Citing Norse
If you use Norse in your work, please cite it as follows:
  ```BibTex
 @software{norse2021,
   author       = {Pehle, Christian and
                   Pedersen, Jens Egholm},
   title        = {{Norse -  A deep learning library for spiking
                    neural networks}},
   month        = jan,
   year         = 2021,
   note         = {Documentation: https://norse.ai/docs/},
   publisher    = {Zenodo},
   version      = {0.0.7},
   doi          = {10.5281/zenodo.4422025},
   url          = {https://doi.org/10.5281/zenodo.4422025}
 }
 ```
Norse is actively applied and cited in the literature. See [Semantic Scholar to view the citations](https://www.semanticscholar.org/paper/Norse-A-deep-learning-library-for-spiking-neural-Pehle-Pedersen/bdd21dfe8c4a503365a49bfdb099e63c74823c7c#citing-papers).

 ## 8. License

 LGPLv3. See [LICENSE](LICENSE) for license details.

