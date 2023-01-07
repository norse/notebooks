---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.4
  kernelspec:
    display_name: Python 3.10.6 64-bit
    language: python
    name: python3
---

# High-Performance Computing with Norse and PyTorch Lightning

![](https://raw.githubusercontent.com/norse/norse-hbp-workshop/main/images/plnorse.png)

Norse provides two necessary requirements for scaling spiking neural network simulations to cluster-wide simulations: a solid infrastructure (PyTorch) and proper state handling. PyTorch permits us to apply a highly developed toolchain (as we will see in a minute) to solve our problems. This saves a dramatic amount of time. Proper state handling permits us to paralellize our simulations, which is practically impossible in many other neuron simulators.

In this small tutorial, you will learn to use Norse with the PyTorch-based framework, [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/latest/) (PL). PL makes it *lightning*-fast (pun intended I'm sure) to build and scale your networks.

The workshop is structured as follows
* Build our first PyTorch Lightning SNN model (~10 minutes)
* Discuss relevant toy problems with your study mates (~5 minutes)
* Try out your own ideas (~15 minutes)


## Step -1: Change Runtime Type

This step is specific to google collab and doesn't apply if you execute this notebook at home or somewhere else. Select "Runtime" above and choose "GPU" as the accelerator. This will make sure
that all of the example code below will run.


## Step 0: Installations

First of all, we will need to install Norse and PyTorch Lightning. Please run the cell below. Read on while it's running.

```python
!pip install norse pytorch-lightning
```

## Step 1: Building our own model

In this part we will get a lightning quick overview of PyTorch Lightning, train a model, and then try to accelerate it. We have little time, so try to just skim over it now and remember that the material is available after the workshop.


### 1.1 What is PyTorch Lightning?

![](https://pytorch-lightning.readthedocs.io/en/stable/_static/images/logo.svg)

The primitives of PyTorch are designed as atoms that we can stitch together to form complicated models. On a more higher level there are some things that become *very* repetitive once you've built your first few models: preprocessing data, defining training loops, measuring loss, plotting your results. This is where PyTorch Lightning comes in. The help us to "Spend more time on research, less on engineering."



### 1.2 PyTorch Lightning models

In vanilla PyTorch we would use the `torch.nn.Module` as a base class. Here, we have to extend the `LightningModule` and implement (at least) three methods:

* `__init__` - The constructor where you build your model
* `configure_optimizers` - This is where you define how your model is optimized
* `training_step` - This is where the model is being applied

```python
import torch
import pytorch_lightning as pl
import norse.torch as norse


class SpikingModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = norse.SequentialState(
            norse.ConstantCurrentLIFEncoder(seq_length=32),  # Encode in time
            norse.ConvNet(),  # Apply convolution
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def training_step(self, batch, batch_index):
        x, y = batch
        # Input has shape (32, 1, 28, 28): (batch, channel, x, y)
        out, state = self.model(x)  # Note the output state; we ignore it for now
        # Output has shape (32, 32, 10) because we encoded each input in 32 timesteps
        # Here we sum up the time dimension and see which class got most spikes
        out, _ = out.max(dim=0)
        return torch.nn.functional.cross_entropy(out, y)
```

Notice that we are not specifiying anything about the data. We are simply assuming that it arrives through the `batch` variable that contains both the input (`x`) and the labels (`y`). Now we simply need to load in the data and start training:


### 1.3 Loading in data

MNIST is a pretty boring example, so that's why we chose it. For now, we just want to get the data loading out of the way so we can start training and scaling our model:

```python
import torchvision

# We need to normalize the data so we get *some* response from the neurons
data_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
    ]
)
dataloader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(root=".", download=True, transform=data_transform),
    batch_size=32,
)
```

### 1.4 Training our model

We are now ready to train our model! The only thing we need is a wrapper class to take care of loading in the data and feeding it to our `LightningModule` module:

```python
model = SpikingModel()  # Our model from before
trainer = pl.Trainer()
trainer.fit(model, dataloader)
```

### 1.5 Help! My model is taking forever!

Yep it is. And this is even just a simple MNIST model! We don't have time to wait for this: go ahead and stop it by pressing the ■ icon above. Notice that PyTorchLightning helpfully warned us that we
have an unused GPU!

Let's try that again, only this time with a GPU:

```python
model = SpikingModel()  # Our model from before
trainer = pl.Trainer(gpus=1)  # Notice the gpus flag
trainer.fit(model, dataloader)
```

<!-- #region -->
### 1.6 Scaling to HPC with PyTorch Lightning

To summarize what we saw so far, we managed to build a model, train it with a dataset, and scale it to 1 (but potentially many) GPUs with around 50 lines of code. Not bad!

You may be wondering where the HPC element in this comes in, but we have actually achieved the most significant objective already: a scalable model. Because Norse is handling state correctly, and because PyTorch Lightning takes care of the synchronization of losses across an arbitrary number of machines, you are able to run this on multiple nodes and several GPUs if you wanted to!

To be slightly more specific, PyTorch Lightning already features support for HPC clusters because your model is easy to scale. The different way to distribute models (e. g. in HPC) is called [`accelerators` in PyTorch Lightning](https://pytorch-lightning.readthedocs.io/en/stable/accelerators.html?highlight=hpc#). Here is an example that will run your model in an HPC (which won't work here for obvious reasons):

```python
model = SpikingModel() # Our model from before
trainer = Trainer(accelerator=DDPHPCAccelerator())
trainer.fit(model, dataloader)
```

You can read more about their [DDPHPCAccelerator here](https://pytorch-lightning.readthedocs.io/en/stable/accelerators.html?highlight=hpc#ddp-hpc-accelerator) or [see how the accelerator parameter works with the Tranier module here](https://pytorch-lightning.readthedocs.io/en/stable/trainer.html#accelerator).
<!-- #endregion -->

## 1.7 SLURM Support out of the Box

You can run your PyTorch-Lightning model with almost no hassle on your favorite SLURM
cluster (JUWELS anyone?) of choice! See https://pytorch-lightning.readthedocs.io/en/latest/slurm.html.
It will helpfully save your progress and restart from the last checkpoint accross
job submissions. You can get your model running accross 2 super computing
nodes and 8 V100 GPU cards without almost no effort of your own.


## Step 2: Discussion

Now that you know how PyTorch works, have seen how Norse builds models, and knows how PyTorch Lightning can accelerate your models, it's time time to discuss how to apply your knowledge!

MNIST is a good example for non-trivial supervised learning. And, as you saw, Norse can indeed solve it.
However, there are much more interesting datasets out there and they might not necessarily be solvable with supervised learning. Norse also supports [STDP](https://github.com/norse/norse/blob/master/norse/torch/functional/stdp.py) (although still in an early stage) and other learning rules.

So, here is our challenge to you: consider your own work/domain/expertise and how spike-based learning (supervised/non-supervised, local/global, etc.) is be relevant. Specifically, ask yourself whether you can come up with neuron-based simulations that are sufficiently large to require HPC access. Could you model this with Norse? Discuss this with your discord group now. Here are some questions you can ask your group to kickstart the discussion:

* Do you think idea X can be solved with spike-based learning?
* How would the dataset look like?
  * Can you learn that with biologically inspired learning algorithms?
  * Would you need supervised learning?
* What learning algorithm would you use to solve X?

When you are done (don't spend more than ~5-10 minutes) move on to the final implementation section.


## Step 3: Implementation

In the final minutes of this workshop, we encourage you to boil down your problem to a really small toy example. You will spend the rest of the workshop trying to implement it, so we recommend that you recruit the aid of your workshop group and try to get something simple working. Please be realistic (in this workshop): we are severely time restricted.

Here are a few recommendations:
* Try to only modify the data and the neural network
  * You can, for instance, modify the code above to solve a simple XOR problem: the network would be much simpler and the data could be generated
* Look at [the examples in the PyTorch Lightning documentation](https://pytorch-lightning.readthedocs.io) for inspiration (in the left sidebar)
* Loot at [the datasets available in Norse](https://github.com/norse/norse/tree/master/norse/dataset). They are event-based so you do not have to worry about encoding


## Step 4: Profit

We only scratched the surface of what PyTorch Lightning provides. They also provide excellent logging/dashboarding with [Tensorboard](https://www.tensorflow.org/tensorboard/), allow model checkpointing, model discretization, and much more.

We hope this was enlightening. The workshop material is available online at https://github.com/norse/norse-hbp-workshop, so you can revisit it any time.

Thank you for your attention!

![](https://raw.githubusercontent.com/norse/norse/master/logo.png)
