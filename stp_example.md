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

# Spike Time Dependent Plasticity

In this tutorial we will take a look at a simple example of plasticity first
experimentally described in a seminal paper by Tsodyks and Makram, which has
been implemented in Norse as an example of a biologically plausible form of
synaptic plasticity.

```python
!pip install norse --quiet
!pip install ipympl --quiet
```

```python
# Install widgets in Google Colab
try:
    from google.colab import output

    output.enable_custom_widget_manager()
except:
    pass
```

```python
import torch
from norse.torch.functional.tsodyks_makram import (
    stp_step,
    TsodyksMakramState,
    TsodyksMakramParameters,
)
```

As a first example consder the case that a single synapse is stimulated by a
periodically firing neuron with a constant frequency of 10 Hertz.

```python
def example(p):
    dt = 0.001
    z = torch.zeros(1000)
    z[::100] = 1.0
    z[0:10] = 0.0

    s = TsodyksMakramState(x=1.0, u=0.0)
    i = 0.0
    xs = []
    us = []
    current = []

    for ts in range(1000):
        x, s = stp_step(z[ts], s, p)

        # integrate the synapse dynamics
        di = -p.tau_s_inv * i
        i = i + dt * di + x
        xs += [s.x]
        us += [s.u]
        current += [i]

    xs = torch.stack(xs)
    us = torch.stack(us)
    current = torch.stack(current)

    return xs, us, current
```

```python
%matplotlib widget
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interact

ts0_fascilitating = TsodyksMakramParameters(
    tau_f_inv=1 / (750.0e-3), tau_s_inv=1 / (20.0e-3), tau_d_inv=1 / (50.0e-3), U=0.15
)

ts0_depressing = TsodyksMakramParameters(
    tau_f_inv=1 / (50.0e-3), tau_s_inv=1 / (20.0e-3), tau_d_inv=1 / (750.0e-3), U=0.45
)

xs, us, current = example(TsodyksMakramParameters())
ts = np.arange(0, 1.0, 0.001)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel("time [s]")
(current_plot,) = ax.plot(ts, current, label="current")
(x_plot,) = ax.plot(ts, xs, label="x")
(u_plot,) = ax.plot(ts, us, label="u")
ax.legend()


@interact(
    tau_f_inv=(0.0, 100.0, 0.5),
    tau_s_inv=(0, 100, 0.5),
    tau_d_inv=(0, 100, 0.5),
    U=(0, 1, 0.1),
)
def update(
    tau_f_inv=1 / (50.0e-3), tau_s_inv=1 / (20.0e-3), tau_d_inv=1 / (750.0e-3), U=0.45
):
    xs, us, current = example(
        TsodyksMakramParameters(tau_f_inv, tau_s_inv, tau_d_inv, U)
    )
    current_plot.set_ydata(current)
    x_plot.set_ydata(xs)
    u_plot.set_ydata(us)
    fig.canvas.draw()
```


```python

```
