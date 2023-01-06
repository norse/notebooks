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
# Training an MNIST classifier

This tutorial introduces the [norse](norse.ai) library by going through the "Hello World" of deep-learning: How to classify hand-written digits. Norse is based on the popular pytorch deep-learning library and this is in fact the only requirement you need to build your own models with it.
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
!pip install --quiet norse
```

<!-- #region id="YrZ71gW94b1O" -->
## Integrating point neuron model equations

Spiking neuron models are given as (typically very simple) systems of ordinary differential
equations. A common example used is the so called current based leaky integrate and fire neuron model (LIF). Its differential equation is given by
\begin{align*}
\dot{v} &= -(v - v_\text{reset}) + I \\
\dot{I} &= -I + I_\text{in}
\end{align*}
together with jump and transition equations, that specify when a jump occurs and
how the state variables change. A prototypical equation is a leaky integrator
with constant current input $I_\text{in}$, with jump condition $v - 1 = 0$ and transition equation $v^+ - v^- = -1$.

<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 279} id="ZzPRyc8E2M8a" outputId="7623e613-923f-451a-f8bf-f1c05ad72912"
from norse.torch.functional import (
    lif_step,
    lift,
    lif_feed_forward_step,
    lif_current_encoder,
    LIFParameters,
)

N = 1  # number of neurons to consider
T = 100  # number of timesteps to integrate

p = LIFParameters()
v = torch.zeros(N)  # initial membrane voltage
input_current = 1.1 * torch.ones(N)

voltages = []

for ts in range(T):
    z, v = lif_current_encoder(input_current, v, p)
    voltages.append(v)

voltages = torch.stack(voltages)
```

We can now plot the voltages over time:

```python
plt.ylabel("v")
plt.xlabel("time [ms]")
plt.plot(voltages)
```

<!-- #region id="MOYVGOvhrDty" -->
## MNIST dataset

A common toy dataset to test machine learning approaches on is the MNIST handwritten digit recognition dataset. The goal is to distinguish handwritten digits 0..9 based on a 28x28 grayscale picture.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 420, "referenced_widgets": ["a39cccabae994485b9b3f0866d6f1891", "48509a5c085441f29d908f03e17104f1", "49e4961d66cf49d096fe542250bd7ea4", "699927de0a9a44438c7f77f53c80ca41", "be29cb04ef804c31acbe5b57a4254f8e", "18f3adf705754ed486b4860c38a443a9", "3313b75ccbea4d1f89409d4abb47b8f6", "d1268b0592cd4f9a9a4c04c939dc0402", "999ca0343fbd4693bfeb6e75df29aaa4", "2537d663e9a941cdb3239daa8f463145", "06c79d8b76d646f597091e64b3ac8a7c", "adf7256611d04523a8d77a6c0079db52", "34055b8bbea048378f7e71a498cba0f8", "eb3b3c6cf9024c7899fbc7f588dec2e0", "d3b69d17635c4be09e1e66d8bf3c190f", "700180bf289944e4898a6ea9bb7e6815", "b38e0b3b846d44f08a68884a6ec894eb", "83d347927a6147f8ae69818bfa66fd88", "58a275f42e63442fa5306f85ee09e094", "77fd86441e54485dad7dfee7b3b26d2b", "89a9086e97834567b65d188efdd0fe66", "dae7cfd7c37f4278b661c9fd14de3aed", "bbd4e89f0ed0472f83ebe824f6177020", "256419cbcfde4ee8bfe6fc35beb236ee", "d4a6f8b8cfc34899a809581d3f10b414", "40735b0849ad472b893f465085013963", "0f11348c44cc48a98135e4e5d2d5de73", "d1aa16b66d21488d93b9b54141190798", "23235a7e41c547f49395137b57387d69", "a4c87414f66747bb9532195c6569364b", "b33e85cc5a304bd798bbd294bae3e236", "50891ba2e81542eeba75ed311f1398ca"]} id="RPs81D-QrFWV" outputId="0eb067e3-fc00-486d-d618-7ca41bd70535"
import torchvision

BATCH_SIZE = 256

transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
    ]
)

train_data = torchvision.datasets.MNIST(
    root=".",
    train=True,
    download=True,
    transform=transform,
)

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(
        root=".",
        train=False,
        transform=transform,
    ),
    batch_size=BATCH_SIZE,
)
```

<!-- #region id="c_STHa4ethi4" -->
## Encoding Input Data

One of the distinguishing features of spiking neural networks is that they
operate on temporal data encoded as spikes. Common datasets in machine learning
of course don't use such an encoding and therefore make a encoding step necessary. Here we choose to treat the grayscale value of an MNIST image
as a constant current to produce input spikes to the rest of the network.
Another option would be to interpret the grayscale value as a spike probabilty
at each timestep.

<!-- #endregion -->

<!-- #region id="pVYrXB00-fqO" -->
### Constant Current Encoder
<!-- #endregion -->

```python id="BHwqgEFvZaoQ"
from norse.torch import ConstantCurrentLIFEncoder
```

<!-- #region id="EHH7pA86l_nL" -->
We can easily visualise the effect of this choice of encoding on a sample image in the training data set
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 292} id="cbYBEbBsTNam" outputId="628bf729-fe32-4d33-de82-a6fcc1257eab"
img, label = train_data[1]

plt.matshow(img[0])
plt.colorbar()
print(label)
```

```python colab={"base_uri": "https://localhost:8080/", "height": 279} id="FWAPfbM7T6lM" outputId="b7b067ea-b534-4017-bcfc-62c33691941b"
T = 32
example_encoder = ConstantCurrentLIFEncoder(T)


example_input = example_encoder(img)
example_spikes = example_input.reshape(T, 28 * 28).to_sparse().coalesce()
t = example_spikes.indices()[0]
n = example_spikes.indices()[1]

plt.scatter(t, n, marker="|", color="black")
plt.ylabel("Input Unit")
plt.xlabel("Time [ms]")
plt.show()
```

<!-- #region id="Ouwn94PQipIw" -->
### Poisson Encoding

As can be seen from the spike raster plot, this kind of encoding does not produce spike patterns which are necessarily biologically realistic. We could rectify this situation by employing cells with varying threshholds and a finer integration time step. Alternatively we can encode the grayscale input images into poisson spike trains



<!-- #endregion -->

```python id="7hqWN47egmir"
from norse.torch import PoissonEncoder
```

<!-- #region id="sOlfpqnBjIrE" -->
This produces a more biological plausible input pattern, as can be seen below:
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 279} id="u74m9sishnkF" outputId="341d9438-236f-46a2-b234-e8edddae8c15"
T = 32
example_encoder = PoissonEncoder(T, f_max=20)

example_input = example_encoder(img)
example_spikes = example_input.reshape(T, 28 * 28).to_sparse().coalesce()
t = example_spikes.indices()[0]
n = example_spikes.indices()[1]

plt.scatter(t, n, marker="|", color="black")
plt.ylabel("Input Unit")
plt.xlabel("Time [ms]")
plt.show()
```

<!-- #region id="Z9QNqiwIx3HH" -->
### Spike Latency Encoding

Yet another example is a spike latency encoder. In this case each input neuron spikes only once, the first time the input crosses the threshhold.
<!-- #endregion -->

```python id="trq4AeDTyKNu"
from norse.torch import SpikeLatencyLIFEncoder
```

```python colab={"base_uri": "https://localhost:8080/", "height": 279} id="JvqFTcvZzfpS" outputId="b1f9eeb0-0b77-4e34-a18e-8f6292dc1dd4"
T = 32
example_encoder = SpikeLatencyLIFEncoder(T)


example_input = example_encoder(img)
example_spikes = example_input.reshape(T, 28 * 28).to_sparse().coalesce()
t = example_spikes.indices()[0]
n = example_spikes.indices()[1]

plt.scatter(t, n, marker="|", color="black")
plt.ylabel("Input Unit")
plt.xlabel("Time [ms]")
plt.show()
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
        self, input_features, hidden_features, output_features, record=False, dt=0.001
    ):
        super(SNN, self).__init__()
        self.l1 = LIFRecurrentCell(
            input_features,
            hidden_features,
            p=LIFParameters(alpha=100, v_th=torch.tensor(0.5)),
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

<!-- #region id="yXzSK17BrmjT" -->
We can visualize the output produced by the recurrent spiking neural network on the example input.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 279} id="nVq-2OD0pSnL" outputId="2d88ed7e-06ec-4424-ea23-8991e2da0d36"
example_snn = SNN(28 * 28, 100, 10, record=True, dt=0.001)

example_readout_voltages = example_snn(example_input.unsqueeze(1))
voltages = example_readout_voltages.squeeze(1).detach().numpy()

plt.plot(voltages)
plt.ylabel("Voltage [a.u.]")
plt.xlabel("Time [ms]")
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 265} id="dbLpIlYqtYJv" outputId="c1b97d2b-b2a6-41eb-f034-ee17f91d1c3c"
plt.plot(example_snn.recording.lif0.v.squeeze(1).detach().numpy())
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/", "height": 265} id="pxK6Ds2Mt94j" outputId="fca1b29f-0e80-4ade-c709-2d8065e4508b"
plt.plot(example_snn.recording.lif0.i.squeeze(1).detach().numpy())
plt.show()
```

<!-- #region id="xq1mvp0ffIsI" -->
## Decoding the Output

The output of the network we have defined are $10$ membrane voltage traces. What remains to do is to interpret those as a probabilty distribution. One way of doing so is to determine the maximum along the time dimension and to then compute the softmax of these values. There are other options of course, for example to consider
the average membrane voltage in a given time window or use a LIF neuron output layer and consider the time to first spike.
<!-- #endregion -->

```python id="6j0wbwEmfbIw"
def decode(x):
    x, _ = torch.max(x, 0)
    log_p_y = torch.nn.functional.log_softmax(x, dim=1)
    return log_p_y
```

<!-- #region id="ngp3zr5AstiH" -->
An alternative way of decoding would be to consider only the membrane trace at the last measured time step.
<!-- #endregion -->

```python id="0Idsu-fAjYdn"
def decode_last(x):
    x = x[-1]
    log_p_y = torch.nn.functional.log_softmax(x, dim=1)
    return log_p_y
```

<!-- #region id="K1jcJ7LnrlUi" -->
## Training the Network

The final model is then simply the sequential composition of these three steps: Encoding, a spiking neural network and decoding.
<!-- #endregion -->

```python id="qRdRp3ZfAYIw"
class Model(torch.nn.Module):
    def __init__(self, encoder, snn, decoder):
        super(Model, self).__init__()
        self.encoder = encoder
        self.snn = snn
        self.decoder = decoder

    def forward(self, x):
        x = self.encoder(x)
        x = self.snn(x)
        log_p_y = self.decoder(x)
        return log_p_y
```

<!-- #region id="vqxyhc_5Xpfg" -->
We can then instantiate the model with the recurrent ```SNN``` network defined above.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="C4QeDXL9_qaB" outputId="533ce0f5-33fe-4cde-897b-218be6d43b9f"
T = 32
LR = 0.002
INPUT_FEATURES = 28 * 28
HIDDEN_FEATURES = 100
OUTPUT_FEATURES = 10

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

model = Model(
    encoder=ConstantCurrentLIFEncoder(
        seq_length=T,
    ),
    snn=SNN(
        input_features=INPUT_FEATURES,
        hidden_features=HIDDEN_FEATURES,
        output_features=OUTPUT_FEATURES,
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

EPOCHS = 5  # Increase this number for better performance


def train(model, device, train_loader, optimizer, epoch, max_epochs):
    model.train()
    losses = []

    for (data, target) in tqdm(train_loader, leave=False):
        data, target = data.to(device), target.to(device)
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
            data, target = data.to(device), target.to(device)
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

for epoch in trange(EPOCHS):
    training_loss, mean_loss = train(
        model, DEVICE, train_loader, optimizer, epoch, max_epochs=EPOCHS
    )
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
trained_snn = model.snn.cpu()
trained_readout_voltages = trained_snn(example_input.unsqueeze(1))
plt.plot(trained_readout_voltages.squeeze(1).detach().numpy())

plt.ylabel("Voltage [a.u.]")
plt.xlabel("Time [ms]")
plt.show()
```

<!-- #region id="RjaJS_Hj2qW4" -->
## Network with Spike Latency Encoding

As we've mentioned above there are alternative ways of encoding and decoding the data to and from spikes. Here we go through two such alternative with the same network we've used before.
<!-- #endregion -->

<!-- #region id="r4sGruNn8w49" -->
As is the the outer training loop.
<!-- #endregion -->

```python id="9EBQxwkW2FEh"
import importlib
from norse.torch.module import encode

encode = importlib.reload(encode)
```

```python colab={"base_uri": "https://localhost:8080/"} id="k0mJvDZx2zRi" outputId="ff5f030b-623f-41ee-e50b-df0f42ccfac2"
# from norse.torch.module import encode

T = 32
LR = 0.002
INPUT_FEATURES = 28 * 28
HIDDEN_FEATURES = 100
OUTPUT_FEATURES = 10

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

model = Model(
    encoder=encode.SpikeLatencyLIFEncoder(T),
    snn=SNN(
        input_features=INPUT_FEATURES,
        hidden_features=HIDDEN_FEATURES,
        output_features=OUTPUT_FEATURES,
    ),
    decoder=decode,
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
model
```

```python colab={"base_uri": "https://localhost:8080/", "height": 83, "referenced_widgets": ["3b858775951f42489a5e99bf258a9579", "efe654038a9f405784dd92ce0fc94641", "32ddf38476e84905b9837a73a2e51c69", "3d9918e3c7124eeea939f5d03f754c2e", "471b51adfd62431eb2cb1cb07e57dfaa", "eeda159d0d8c4f68a5072c61c8a148ec", "088c594b5d9a4d2fb7fe40aaece3715f", "c43606fcb05742bdab65e48b89160e23", "d0b06109fe624b7b89f254d271501421", "85dd1918d25e49e68d7fa33098b1a944", "1dcdbfd15aee439094c6629293a29ad2", "eb527019b41646298444cbd2d41fd61d", "3ef4a9dccdfe49ecb66766a2a35e609b", "401931c694c54d65af8ee39770c10725", "7607a137be384342b4477d8c2aa3c872", "cab564591c9c412b9e6e1b82bb5bdabb", "5dc2ae41b0fe44eb8a54dbfda5ad3662", "2bb875b41fdc4805943dff47a5213970", "69c0a6d00ebf47c58ea3b96fc7e57b05", "22564260e03e4c89a5b65696f090e101", "2890f00690c64ab3b1c2854c64a654f7", "8f6a1df4fda548869e1326a09868b3dc", "a9edd88e94544116a5187e0424b0c177", "6af440e96d7f4bc3b629029f237fe13b", "8327d4dcf9a243d090e99593fb6e2fd5", "a46a756afa2841e4867d4b9bf99482f4", "22d37cddba6347a3b8b2909dbce03fd5", "1c8d9211202a49eaa066eecba1fc933b", "683c5718d3bd471693ed66eec49bcdd3", "14dfca545dd24dc68b1062bedc704233", "0404b45b19884af6beb843d18e1b7ba7", "6e8be9fab26449e9a43b8005fb6f4ac5", "92c8bdb639ae4cecba7d02f36ab029ec", "05821971a2b44b29ba6d4845c292557f", "fda71ad25ca848429c1db23b866f16b3", "39a76cf8e4bb42f1b6132299dd8ac19e", "2a47df900af5446aa4a9221f273f76d6", "11fb619891d84f52a3324729dad1d25e", "dc6d60ac28be4447b284b3f132fcd029", "f7452107d5484825b40393f03f72a5fd", "c23f52e042654b7b9c2c6fe9105e2b3d", "7b784beef6104c2380fe70afaebc0205", "de6875ef3b6f4869acb712ca2db6a772", "c30319363add4b2aba55271ed6904bce", "9aee06638dbe495faea380dd1970d6fe", "1ebf55899c5d4cca97f0efe349e9d770", "8f308f906ea0445bad9543048f69ce41", "4f29018687b844e1beaf6cc6a7198ad2"]} id="jynkfxjr4maa" outputId="45aa79eb-5e86-4325-9b10-40cce6000125"
training_losses = []
mean_losses = []
test_losses = []
accuracies = []

for epoch in trange(EPOCHS):
    training_loss, mean_loss = train(
        model, DEVICE, train_loader, optimizer, epoch, max_epochs=EPOCHS
    )
    test_loss, accuracy = test(model, DEVICE, test_loader, epoch)
    training_losses += training_loss
    mean_losses.append(mean_loss)
    test_losses.append(test_loss)
    accuracies.append(accuracy)

print(f"final accuracy: {accuracies[-1]}")
```

<!-- #region id="jUQSr3GAQ_Wd" -->
## Network with Poisson Encoded Input


<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/"} id="Hx7vMD2SQ51V" outputId="b79ebdab-deea-4ac5-e5f4-8d714424a97e"
T = 32
LR = 0.002
INPUT_FEATURES = 28 * 28
HIDDEN_FEATURES = 100
OUTPUT_FEATURES = 10

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

model = Model(
    encoder=encode.PoissonEncoder(T, f_max=20),
    snn=SNN(
        input_features=INPUT_FEATURES,
        hidden_features=HIDDEN_FEATURES,
        output_features=OUTPUT_FEATURES,
    ),
    decoder=decode,
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
model
```

```python colab={"base_uri": "https://localhost:8080/", "height": 83, "referenced_widgets": ["990ed923e0c347958bce4ceeaa5484e0", "6c9442f6e98848078f4979473d10222e", "6c5f798c7342485f95287dc77a89119a", "d1f3a0ed896a486c8eba7cbfb929d036", "1b73683e800e4691ad6b38b3acb61b27", "ed6630afa2cf41f49bd26be65f103182", "08dd6a2ceeb3472cb574fd9fefb62e07", "0d9a635181f94dfe9e248b9f4902f0f8", "bfbd187ff861471d90448b5e805336f6", "84e7f2f4891f4c91ba0a8d8469d16b20", "971f5a7ac6f84271b90f042993681eed", "1fa7599ee6094a8c8e96a3d3d0c571f4", "c1cbd84429004c0085990ab4345a4767", "9129f0f4435147b38733be723bb940f5", "c051e4af46394ea6b04201c66071a9af", "ff09ad454d3f4ccbb27da78303ede1f9", "47a36f4376a7438d874fd74e66904407", "4c2139032dca4926a82dab6b16cfe9df", "69e979a4a4804e02baae9bb771fd58cd", "d930494b8fa1477288028042602f8bfa", "e49d0a23508941d3a66509c7e4ab1542", "be2c349fc768445c807efb9e6f234322", "ea141499b9e1445485417368364afc4b", "7b22a159c0f441488d9371e1268fe2eb", "d83ceb619ffe4de193b373b46785b9c5", "21ed7ad16b28448f9e9574b0ead5cbb1", "1a6a79f3ef22469a94d0c158518aaea2", "69d0ac9eb51e46f1b47b51d59c61b407", "b560d6a6173d426b82b779491e28033e", "42124470a0584e02b67c99d90bb237dd", "f9b5a586a44f4c6fa844575945552988", "6106151e6553435f963ae04e5b1784d0", "eee26a654fc240bf9ec3cab72e55808e", "dfc0b48867314fa3824e08f538924d0f", "378fa6ee5309489b8f5a7e3e45b7a955", "1fbe4bce97ba4016a4281f79ee812524", "f019ff0adcc04ae7ae6dff53e25adf45", "ae2c7cc3c7014c57a897adb095947cce", "a98a4e0a6654438fb884ded46862f12f", "5ae6dfc79b6d4b41a3ae4dbe151897ba", "52a67a8e585e43f7b880c0a7ccaa9bc9", "34bf1fb8465b41b7b856ebc1218e4f18", "966260f6138e40b98a467c2228ff3e50", "85b1a74aa55c4a6aac158560122006d5", "17f1c4e1a91e4b8bb1ce662feda382b1", "c0dc9f0d5c424d46b305aa31852c38bd", "9300acf406334c4286cdfab47f737c45", "7e9cd29064d948e8888bc200dde0b30c"]} id="9u1HP4LTRnQF" outputId="5d919db4-c531-4cf2-f5b4-c497f728d008"
training_losses = []
mean_losses = []
test_losses = []
accuracies = []

for epoch in trange(EPOCHS):
    training_loss, mean_loss = train(
        model, DEVICE, train_loader, optimizer, epoch, max_epochs=EPOCHS
    )
    test_loss, accuracy = test(model, DEVICE, test_loader, epoch)
    training_losses += training_loss
    mean_losses.append(mean_loss)
    test_losses.append(test_loss)
    accuracies.append(accuracy)
    # print(f"epoch: {epoch}, mean_loss: {mean_loss}, test_loss: {test_loss}, accuracy: {accuracy}", flush=True)

print(f"final accuracy: {accuracies[-1]}")
```

<!-- #region id="YFLeAQPrkNbi" -->
As can be seen from the training result, this combination of hyperparameters, decoding and encoding scheme performs worse than the alternative we've presented before. As with any machine learning approach one of the biggest challenges is to find a combination of these choices that works well. Sometimes theoretical knowledge helps in making these choices. For example it is well known that poisson encoded input will converge with $1/\sqrt{T}$, where $T$ is the number of timesteps. So most likely the low number of timesteps ($T = 32$) contributes to the poor performance.

In the next section we will see that choice of network architecture is also key in training performant spiking neural networks, just as it is for artifiicial neural networks.
<!-- #endregion -->

<!-- #region id="V2iFlyC-r40a" -->
## Convolutional Networks

The simple two layer recurrent spiking neural network we've defined above achieves a respectable ~96.5% accuracy after 10 training epochs. One common way
to improve on this performance is to use convolutional neural networks. We define here two convolutional layers and one spiking classification layer. Just as in the recurrent spiking neural network before, we use a non-spiking leaky integrator for readout.

The ```torch.nn.functional.max_pool2d``` on binary values is a logical ```or``` operation on its inputs.

<!-- #endregion -->

```python id="BEcU9GMdBc8r"
from norse.torch.module.leaky_integrator import LILinearCell
from norse.torch.functional.lif import LIFFeedForwardState
from norse.torch.functional.leaky_integrator import LIState

from typing import NamedTuple


class ConvNet(torch.nn.Module):
    def __init__(self, num_channels=1, feature_size=28, method="super", alpha=100):
        super(ConvNet, self).__init__()

        self.features = int(((feature_size - 4) / 2 - 4) / 2)

        self.conv1 = torch.nn.Conv2d(num_channels, 20, 5, 1)
        self.conv2 = torch.nn.Conv2d(20, 50, 5, 1)
        self.fc1 = torch.nn.Linear(self.features * self.features * 50, 500)
        self.lif0 = LIFCell(p=LIFParameters(method=method, alpha=alpha))
        self.lif1 = LIFCell(p=LIFParameters(method=method, alpha=alpha))
        self.lif2 = LIFCell(p=LIFParameters(method=method, alpha=alpha))
        self.out = LILinearCell(500, 10)

    def forward(self, x):
        seq_length = x.shape[0]
        batch_size = x.shape[1]

        # specify the initial states
        s0 = s1 = s2 = so = None

        voltages = torch.zeros(
            seq_length, batch_size, 10, device=x.device, dtype=x.dtype
        )

        for ts in range(seq_length):
            z = self.conv1(x[ts, :])
            z, s0 = self.lif0(z, s0)
            z = torch.nn.functional.max_pool2d(z, 2, 2)
            z = 10 * self.conv2(z)
            z, s1 = self.lif1(z, s1)
            z = torch.nn.functional.max_pool2d(z, 2, 2)
            z = z.view(-1, 4**2 * 50)
            z = self.fc1(z)
            z, s2 = self.lif2(z, s2)
            v, so = self.out(torch.nn.functional.relu(z), so)
            voltages[ts, :, :] = v
        return voltages
```

```python colab={"base_uri": "https://localhost:8080/", "height": 292} id="dWuEYbSK6BVc" outputId="0d2b5141-d91b-44ff-8d3e-59832ea2d3a3"
img, label = train_data[2]

plt.matshow(img[0])
plt.show()
print(label)
```

<!-- #region id="oiIMzaks6wVp" -->
Just as we did we can visualise the output of the untrained convolutional network on a sample input. Notice that compared to the previous untrained
output the first non-zero membrane trace values appear later. This is due to
the fact that there is a finite delay for each added layer in the network.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 279} id="ej6ADCz7zYNj" outputId="ff8a6e9f-df2b-4052-d8c4-3ff508ffe231"
T = 48
example_encoder = encode.ConstantCurrentLIFEncoder(T)
example_input = example_encoder(img)
example_snn = ConvNet()
example_readout_voltages = example_snn(example_input.unsqueeze(1))

plt.plot(example_readout_voltages.squeeze(1).detach().numpy())
plt.ylabel("Voltage [a.u.]")
plt.xlabel("Time [ms]")
plt.show()
```

```python colab={"base_uri": "https://localhost:8080/"} id="F0ZP9KzzUKiG" outputId="087898c3-bfef-4dda-a037-9bb79c0f580f"
T = 48
LR = 0.001
EPOCHS = 5  # Increase this for improved accuracy

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

model = Model(
    encoder=encode.ConstantCurrentLIFEncoder(T), snn=ConvNet(alpha=80), decoder=decode
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
model
```

```python colab={"base_uri": "https://localhost:8080/", "height": 83, "referenced_widgets": ["ad67a7f23c9e452a932a1fc9128a7190", "50a7821a77824429b29fed56bc3818ae", "91b005a8c65647fb8eb61c18cebafc03", "24f89a01ee774a6d801cb35e3c6e9193", "747d08455ba74fe09a719a982addbe42", "860bebe74a194db3a3635a5311294706", "a00debf1db644d5e9fdacc226434676c", "aa416f4f7839451e9a0ec68277278a19", "0eb9acebcb6a4c54b50718506ba5e270", "6d9601fa6d834534a6124421c60862aa", "525d5c3e5f0c45c883706b06739301bb", "f0a9d5f9ec0348c3a5dbb497b1300318", "99d25e55cda04ff1a7c19e91393409c4", "ffdaac8f4a6246468cdf1b0ae7e287a7", "8e57797683134cf99afe3e7e368cce75", "32d93c0d270f40ebbc384d8ed3b06240", "6fe40cc40b67481ca772efa707102250", "e4c7581f302843c9a9cdec0893cc6e0b", "651ca62a948e435d82f76faac71c717d", "79718e60cdfc43e6b810639581520cff", "059e662d13164f4db032a3c2a2e68d7e", "3da54ed82b004184b4053da8387cd4d9", "424b45fb613f440ab88786c9594b42dc", "e6aa0c3139a646f2b8f68125a4816aca", "f66c2bd6087745578d46f09a6c424554", "9e247f32d7454a9db2431fd75b22d583", "178049a186db43bd960f7b16c92fc8cf", "cc00703fde694eab899cf40f72b854bb", "f561c7182c0c4534ae890fc81b19dabe", "4b87384f7913442b81fbae1bd7aedd86", "1437267691404da0a7aeb3b087ae989a", "dec0cccb5d5c4a7281b71151d4b0ac7f", "cf04ca2f8d0441088520d6e463c7e2c3", "0a7f646b3fc6457799100371f8870c04", "b821a47bb7ef40a6b2f3e0fed62213af", "1438cb1be47842ea87ff2ec8fc0ffdac", "c1a451f2d31f46e2bcd36a68180308bc", "55e802c92c6443bd9ea90ff62d641deb", "591f49916bc349a19dbb3365b8d2b046", "ee4e5845ab2f4f07ba11dfc9d1f73ec9", "dd75cfad6f4f40a687e0fdfde9ac22de", "88d4880a1e494778ad5e356e8680f323", "199c50a26d4d4d4e9d6cd6bddb07db69", "e4a5a0aed7a24145a18fcf3233316657", "8a94aa7fde084b7a9a434043dc648381", "23ef32e7c25946c9918008978eb836ca", "130b66bce2a54c19bddc22802ea2bc9d", "063c92c1db58422f9069a750e2f6dbc0"]} id="IL46G6sdVxoN" outputId="43f9c98e-3d95-4a56-bdca-8b1cf76370d2"
training_losses = []
mean_losses = []
test_losses = []
accuracies = []

for epoch in trange(EPOCHS):
    training_loss, mean_loss = train(
        model, DEVICE, train_loader, optimizer, epoch, max_epochs=EPOCHS
    )
    test_loss, accuracy = test(model, DEVICE, test_loader, epoch)
    training_losses += training_loss
    mean_losses.append(mean_loss)
    test_losses.append(test_loss)
    accuracies.append(accuracy)

print(f"final accuracy: {accuracies[-1]}")
```

```python colab={"base_uri": "https://localhost:8080/", "height": 298} id="R6aZXCqOVZjI" outputId="1e8d91b1-9d7f-4fee-abcc-f664cb156a20"
trained_snn = model.snn.cpu()

trained_readout_voltages = trained_snn(example_input.unsqueeze(1))

print(trained_readout_voltages.shape)

for i in range(10):
    plt.plot(
        trained_readout_voltages[:, :, i].squeeze(1).detach().numpy(), label=f"{i}"
    )

plt.ylabel("Voltage [a.u.]")
plt.xlabel("Time [ms]")
plt.legend()
plt.show()
```

<!-- #region id="4YKMRqTOfO9l" -->
As we can see the output neuron for the label '4' indeed integrates
the largest number of spikes.
<!-- #endregion -->

```python colab={"base_uri": "https://localhost:8080/", "height": 292} id="pRjmCTIdcawn" outputId="d56e73b0-22df-4162-eb4f-096763dc317b"
plt.matshow(np.squeeze(img, 0))
```

<!-- #region id="M_OUx3IKlQLE" -->
## Conclusions

We've seen that on a small supervised learning task it is relatively easy to define spiking neural networks that perform about as well as non-spiking artificial networks. The network architecture used is in direct correspondence to one that would be used to solve such a task with an artificial neural network, with the non-linearities replaced by spiking units.

The remaining difference in performance might be related to a number of choices:
- hyperparameters of the optimizer
- precise architecture (e.g. dimensionality of the classification layer)
- weight initialisation
- decoding scheme
- encoding scheme
- number of integration timesteps

The first three points are in common with the problems encountered in the design and training of artificial neural network classifiers. Comparatively little is known though about their interplay for spiking neural network architectures.

The last three points are special to spiking neural network problems simply because of their constraints on what kind of data they can process naturally. While their interplay has certainly been investigated in the literature, it is unclear if there is a good answer what encoding and decoding should be chosen in general.

Finally we've also omitted any regularisation or data-augementation, which could further improve performance. Common techniques would be to introduce weight decay or penalise unbiologically high firing rates. In the simplest case those can enter as addtional terms in the loss function we've defined above.
<!-- #endregion -->
