---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.4
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region id="b7IYU0Bqomb2" -->
# Training a classifier on the event-based POKER-DVS dataset

When working with Spiking Neural Networks (SNN), we will inevitably encounter the notion of _time_ in our network and data flow. The classic example of MNIST handwritten digits consists of images, much like snapshots in time. Deep learning has shown impressive results on such purely spatial compositions, but SNNs might be able to extract meaning from temporal features and/or save power doing so in comparison to classical networks.

An event camera such as the Dynamic Vision Sensor (DVS) is [somewhat based](https://medium.com/@gregorlenz/rethinking-the-way-our-cameras-see-8584b5167bb) on the functional principle of the human retina. Such a camera can record a scene much more efficiently than a conventional camera by encoding the changes in a visual scene rather than absolute illuminance values. The output is a spike train of change detection events for each pixel. While previously we had to use encoders to equip static image data with a temporal dimension, the POKER-DVS dataset contains recordings of poker cards that are shown to an event camera in rapid succession.

**Warning!** This notebook uses a large dataset and can take a significant amount of time to execute.
<!-- #endregion -->

```python id="wu93JGgT2CJ2"
import torch
import numpy as np
import matplotlib.pyplot as plt
```

<!-- #region id="9rmUJSdzqypr" -->
We can simply install Norse through pip:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="DPb7tCeX2Jkb" outputId="27c75437-737b-43f3-eb76-10ec7e7983ad"
!pip install norse --quiet
```

For this tutorial we are going to make use of a package that handles event-based datasets called [Tonic](https://github.com/neuromorphs/tonic). It is based on PyTorch Vision, so you should already have most of its dependencies installed.

```python
!pip install tonic --quiet
```

Let's start by loading the POKER-DVS dataset and specifying a sparse tensor transform whenever a new sample is loaded

```python colab={"base_uri": "https://localhost:8080/", "height": 279} id="ZzPRyc8E2M8a" outputId="7623e613-923f-451a-f8bf-f1c05ad72912"
import tonic
import torchvision

sensor_size = tonic.datasets.POKERDVS.sensor_size
frame_transform = tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=1000)

trainset = tonic.datasets.POKERDVS(save_to="./data", train=True)
testset = tonic.datasets.POKERDVS(
    save_to="./data", transform=frame_transform, train=False
)
```

We can have a look at how a sample of one digit looks like. The event camera's output is encoded as events that have x/y coordinates, a timestamp and a polarity that indicates whether the lighting increased or decreased at that event. The events are provided in an (NxE) array. Let's have a look at the first example in the dataset. Every row in the array represents one event of timestamp, x, y, and polarity.

```python
events = trainset[0][0]
events
```

When accumulated over time into 3 bins, the images show 1 of 4 card symbols

```python
tonic.utils.plot_event_grid(events)
```

And this one is the target class:

```python
trainset[0][1]
```

We wrap the training and testing sets in PyTorch DataLoaders that facilitate file loading. Note also the custom collate function __pad_tensors__ , which makes sure that all sparse tensors in the batch have the same dimensions

```python colab={"base_uri": "https://localhost:8080/", "height": 420, "referenced_widgets": ["a39cccabae994485b9b3f0866d6f1891", "48509a5c085441f29d908f03e17104f1", "49e4961d66cf49d096fe542250bd7ea4", "699927de0a9a44438c7f77f53c80ca41", "be29cb04ef804c31acbe5b57a4254f8e", "18f3adf705754ed486b4860c38a443a9", "3313b75ccbea4d1f89409d4abb47b8f6", "d1268b0592cd4f9a9a4c04c939dc0402", "999ca0343fbd4693bfeb6e75df29aaa4", "2537d663e9a941cdb3239daa8f463145", "06c79d8b76d646f597091e64b3ac8a7c", "adf7256611d04523a8d77a6c0079db52", "34055b8bbea048378f7e71a498cba0f8", "eb3b3c6cf9024c7899fbc7f588dec2e0", "d3b69d17635c4be09e1e66d8bf3c190f", "700180bf289944e4898a6ea9bb7e6815", "b38e0b3b846d44f08a68884a6ec894eb", "83d347927a6147f8ae69818bfa66fd88", "58a275f42e63442fa5306f85ee09e094", "77fd86441e54485dad7dfee7b3b26d2b", "89a9086e97834567b65d188efdd0fe66", "dae7cfd7c37f4278b661c9fd14de3aed", "bbd4e89f0ed0472f83ebe824f6177020", "256419cbcfde4ee8bfe6fc35beb236ee", "d4a6f8b8cfc34899a809581d3f10b414", "40735b0849ad472b893f465085013963", "0f11348c44cc48a98135e4e5d2d5de73", "d1aa16b66d21488d93b9b54141190798", "23235a7e41c547f49395137b57387d69", "a4c87414f66747bb9532195c6569364b", "b33e85cc5a304bd798bbd294bae3e236", "50891ba2e81542eeba75ed311f1398ca"]} id="RPs81D-QrFWV" outputId="0eb067e3-fc00-486d-d618-7ca41bd70535"
# reduce this number if you run out of GPU memory
BATCH_SIZE = 32

# add sparse transform to trainset, previously omitted because we wanted to look at raw events
trainset.transform = frame_transform

train_loader = torch.utils.data.DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    collate_fn=tonic.collation.PadTensors(batch_first=False),
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    collate_fn=tonic.collation.PadTensors(batch_first=False),
    shuffle=False,
)
```

<!-- #region id="cNEqcSNH2WfP" -->
## Defining a Network

Once the data is encoded into spikes, a spiking neural network can be constructed in the same way as a one would construct a recurrent neural network.
Here we define a spiking neural network with one recurrently connected layer
with `hidden_features` LIF neurons and a readout layer with `output_features` and leaky-integrators. As you can see, we can freely combine spiking neural network primitives with ordinary `torch.nn.Module` layers.
<!-- #endregion -->

```python id="qN9Sm4rJtgc7"
from norse.torch import LIFParameters, LIFState
from norse.torch.module.lif import LIFCell, LIFRecurrentCell

# Notice the difference between "LIF" (leaky integrate-and-fire) and "LI" (leaky integrator)
from norse.torch import LICell, LIState

from typing import NamedTuple


class SNNState(NamedTuple):
    lif0: LIFState
    readout: LIState


class SNN(torch.nn.Module):
    def __init__(
        self,
        input_features,
        hidden_features,
        output_features,
        tau_syn_inv,
        tau_mem_inv,
        record=False,
        dt=1e-3,
    ):
        super(SNN, self).__init__()
        self.l1 = LIFRecurrentCell(
            input_features,
            hidden_features,
            p=LIFParameters(
                alpha=100,
                v_th=torch.as_tensor(0.3),
                tau_syn_inv=tau_syn_inv,
                tau_mem_inv=tau_mem_inv,
            ),
            dt=dt,
        )
        self.input_features = input_features
        self.fc_out = torch.nn.Linear(hidden_features, output_features, bias=False)
        self.out = LICell(dt=dt)

        self.hidden_features = hidden_features
        self.output_features = output_features
        self.record = record

    def forward(self, x):
        seq_length, batch_size, _, _, _ = x.shape
        s1 = so = None
        voltages = []

        if self.record:
            self.recording = SNNState(
                LIFState(
                    z=torch.zeros(seq_length, batch_size, self.hidden_features),
                    v=torch.zeros(seq_length, batch_size, self.hidden_features),
                    i=torch.zeros(seq_length, batch_size, self.hidden_features),
                ),
                LIState(
                    v=torch.zeros(seq_length, batch_size, self.output_features),
                    i=torch.zeros(seq_length, batch_size, self.output_features),
                ),
            )

        for ts in range(seq_length):
            z = x[ts, :, :, :].view(-1, self.input_features)
            z, s1 = self.l1(z, s1)
            z = self.fc_out(z)
            vo, so = self.out(z, so)
            if self.record:
                self.recording.lif0.z[ts, :] = s1.z
                self.recording.lif0.v[ts, :] = s1.v
                self.recording.lif0.i[ts, :] = s1.i
                self.recording.readout.v[ts, :] = so.v
                self.recording.readout.i[ts, :] = so.i
            voltages += [vo]

        return torch.stack(voltages)
```

It's a good idea to test the network's response to time constant parameters that depend on the duration of recordings in the dataset as well as average number of events. We use dt=1e-6 because the events we're dealing with have microsecond resolution

```python
example_snn = SNN(
    np.product(trainset.sensor_size),
    100,
    len(trainset.classes),
    tau_syn_inv=torch.tensor(1 / 1e-2),
    tau_mem_inv=torch.tensor(1 / 1e-2),
    record=True,
    dt=1e-3,
)

frames, target = next(iter(train_loader))

frames[:, :1].shape
```

Note that we are only applying a subset (`1000`) of the data timesteps (`22227`).

```python
example_readout_voltages = example_snn(frames[:, :1])
voltages = example_readout_voltages.squeeze(1).detach().numpy()

plt.plot(voltages)
plt.ylabel("Voltage [a.u.]")
plt.xlabel("Time [us]")
plt.show()
```

```python
plt.plot(example_snn.recording.lif0.v.squeeze(1).detach().numpy())
plt.show()
```

```python
plt.plot(example_snn.recording.lif0.i.squeeze(1).detach().numpy())
plt.show()
```

<!-- #region id="K1jcJ7LnrlUi" -->
## Training the Network

The final model is then simply the sequential composition of our network and a decoding step.
<!-- #endregion -->

```python
def decode(x):
    x, _ = torch.max(x, 0)
    log_p_y = torch.nn.functional.log_softmax(x, dim=1)
    return log_p_y


class Model(torch.nn.Module):
    def __init__(self, snn, decoder):
        super(Model, self).__init__()
        self.snn = snn
        self.decoder = decoder

    def forward(self, x):
        x = self.snn(x)
        log_p_y = self.decoder(x)
        return log_p_y
```

```python colab={"base_uri": "https://localhost:8080/"} id="C4QeDXL9_qaB" outputId="533ce0f5-33fe-4cde-897b-218be6d43b9f"
LR = 0.002
INPUT_FEATURES = np.product(trainset.sensor_size)
HIDDEN_FEATURES = 100
OUTPUT_FEATURES = len(trainset.classes)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

model = Model(
    snn=SNN(
        input_features=INPUT_FEATURES,
        hidden_features=HIDDEN_FEATURES,
        output_features=OUTPUT_FEATURES,
        tau_syn_inv=torch.tensor(1 / 1e-2),
        tau_mem_inv=torch.tensor(1 / 1e-2),
    ),
    decoder=decode,
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
model
```

<!-- #region id="rM5btRjKdEEv" -->
What remains to do is to setup training and test code. This code is completely independent of the fact that we are training a spiking neural network and in fact has been largely copied from the pytorch tutorials.
<!-- #endregion -->

```python id="SXgntmL_rvHO"
from tqdm.notebook import tqdm, trange


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    losses = []

    for (data, target) in tqdm(train_loader, leave=False):
        data, target = data.to(device), torch.LongTensor(target).to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    mean_loss = np.mean(losses)
    return losses, mean_loss
```

<!-- #region id="KtdcQi_18Xip" -->
Just like the training function, the test function is standard boilerplate, common with any other supervised learning task.
<!-- #endregion -->

```python id="Gca4ZzatApWD"
def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), torch.LongTensor(target).to(device)
            output = model(data)
            test_loss += torch.nn.functional.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    accuracy = 100.0 * correct / len(test_loader.dataset)

    return test_loss, accuracy
```

```python colab={"base_uri": "https://localhost:8080/", "height": 66, "referenced_widgets": ["83471770562e4dc6a9c0203567a11738", "79a47e5c73884eaa95aa3cf1798bd34e", "e8815d2923f64385993f68ba2a0ff83a", "c9dddaf73fac46d78a2616763c3498ff", "a51b7d3fd61d450c89bc5525ce31c495", "9904c1cbdddf479a8491807d3810b4a0", "c65e235f0f21418b80d8df61fcea0229", "1cec1ba4c03342f29d6fa3bac1fdef79", "0db2860f929b4728a58a5a60a76d0661", "afe0f290c061434a9868f4f7884ecae4", "5a4f2f4068e3463fbcec7a8cfcc6d99c", "a4cbd02d8b47464a8fefab8bd4be1a4f", "1c76c1d0a7a443b89e34a6606b5db3e2", "6ffc4204c106450c82577624db006d7d", "ecce7c70c4bb41cb9984529e1ea836d9", "6a0653febe6443768d9f04693a6815dc", "ba0526a00832430696c50c3c3990ea52", "37ff024648ef4a328ab218827d1e5e8a", "f2581d6f8e014b7f8c47c9748966ddc5", "5f5c829842734e109a062d7822308738", "d307e64e0add463e8ec0fb8556c02532", "63d12d053b194af5a064c23916544fe6", "fdcd12c58ce34ac9a1d64132e9dfaade", "528b36c345eb4c16a043b3fcb0d26a02", "f82b17289112486fa1340b5d7dd14c0e", "dc0e7177ec804b0fb99b85d56eaf3ad5", "4f77245233e54f01a23d2070d049f63d", "ec712d288cee4f1eb2bcf6af2a2e3673", "ae87af3bb7014b19b0dd0c515a115724", "ee324024183546d5b5e563ceb5bcbd6e", "71f7be117e3840ed8ecd71c30b92ccd1", "0d985ad92a734753923318035caee6d8", "4e9e3649f45a4fd78cff0efb1a776c8d", "08d8223656ea4402851911a5b8b320b6", "120e2e6d1209454bac8a77dba77a3181", "983c4760eb4247f2b053b70fa8c470f8", "235c4ed216ef4bebbe5cbb45e5e8a432", "4e81dcf2a69e4255b68f96594a5d8f46", "e3233a8ac4b548b78d62f5312a624b39", "4690821d1e4744ce8a1e00bd92f612dd", "a01d369fee314a3390ee34d4c7c27c63", "21d8540ae0a5444f8cdca7d00d3b3034", "7bab9f7eead449d6b2f043fddeaeaa5f", "ef6ff21e1b8843939c97d844d8245330", "f239ddb5cf8a491d857d4e1bbcb1bc06", "cd0f7ad4e21f4b2580f9dd0c8d333586", "5adc970dab4346a9a54afbeeb81c91e0", "34988d7cfbbb42aeac37b7dfe5fc1288"]} id="wU-b7Q8eBVca" outputId="5f992877-857f-4630-c9ae-78ee999679bf"
training_losses = []
mean_losses = []
test_losses = []
accuracies = []

torch.autograd.set_detect_anomaly(True)

EPOCHS = 10

for epoch in trange(EPOCHS):
    training_loss, mean_loss = train(model, DEVICE, train_loader, optimizer, epoch)
    test_loss, accuracy = test(model, DEVICE, test_loader, epoch)
    training_losses += training_loss
    mean_losses.append(mean_loss)
    test_losses.append(test_loss)
    accuracies.append(accuracy)

print(f"final accuracy: {accuracies[-1]}")
```

<!-- #region id="XKEVGF76x_Ee" -->
We can visualize the output of the trained network on an example input
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 279} id="NL_rtLC2xXLp" outputId="f6d08cf8-d3f1-4636-f190-ed6421d29256"
trained_snn = model.snn
trained_readout_voltages = trained_snn(frames[:, :1].to("cuda"))
plt.plot(trained_readout_voltages.squeeze(1).cpu().detach().numpy())

plt.ylabel("Voltage [a.u.]")
plt.xlabel("Time [ms]")
plt.show()
```

```python

```
