import os, json

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.evaluation import all_activity, assign_labels, proportion_weighting
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.manual_seed_all(0)
torch.set_num_threads(os.cpu_count() - 1)

import IPython
import numpy as np
from IPython.display import display, HTML, Javascript
import time
import random

# google colab plotting
def configure_browser_state():
  display(IPython.core.display.HTML('''
    <canvas id="myChart"></canvas>
  '''))
  display(IPython.core.display.HTML('''
        <script src="https://cdn.jsdelivr.net/npm/chart.js@2.8.0"></script>
        <script>
          var ctx = document.getElementById('myChart').getContext('2d');
          var chart = new Chart(ctx, {
              // The type of chart we want to create
              type: 'line',

              // The data for our dataset
              data: {
                  labels: [0,1,2,3,4,5],
                  datasets: [{
                      label: 'Score',
                      borderColor: 'rgb(255, 99, 132)',
                      data: [0,1,2,3,4,5]
                  }]
              },

              // Configuration options go here
              options: {
                animation: {
                  duration: 0, // general animation time
                }
              }
          });

          function addData(label, value){
            chart.data.labels.push(label)
            chart.data.datasets[0].data.push(value)

            // optional windowing
            if(chart.data.labels.length > 10) {
              chart.data.labels.shift()
              chart.data.datasets[0].data.shift()
            }

            chart.update();
          }
        </script>
        '''))



def load_config(filename):
    with open(filename) as f:
        config = json.load(f)
    return config

def create_network(n_neurons, exc, inh, dt, theta_plus, lr) -> DiehlAndCook2015:
    network = DiehlAndCook2015(
        n_inpt=784,
        n_neurons=n_neurons,
        exc=exc,
        inh=inh,
        dt=dt,
        nu=lr,  # 0.711
        norm=78.4,
        theta_plus=theta_plus,
        inpt_shape=(1, 28, 28),
    )
    network.to("cuda")

    return network

def create_monitorings(network: DiehlAndCook2015, time: int):
    spikes = {}
    for layer in set(network.layers):
        spikes[layer] = Monitor(network.layers[layer], state_vars=["s"], time=time)
        network.add_monitor(spikes[layer], name=f"{layer}_spikes")

    return network, spikes

def train_network(network, dataset, spikes, n_train, update_interval, n_classes, time, n_neurons, plot=False):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    spike_record = torch.zeros(update_interval, time, n_neurons, device=device)
    accuracy = {"all": [], "proportion": []}
    labels = torch.empty(update_interval, device=device)
    per_class = int(n_neurons / n_classes)

    assignments = -torch.ones_like(torch.Tensor(n_neurons), device=device)
    proportions = torch.zeros_like(torch.Tensor(n_neurons, n_classes), device=device)
    rates = torch.zeros_like(torch.Tensor(n_neurons, n_classes), device=device)
    # Train the network.
    print("Begin training.\n")

    pbar = None
    if not plot:
        pbar = tqdm(total=n_train)
    for (i, datum) in enumerate(dataloader):
        if i > n_train:
            break

        image = datum["encoded_image"]
        label = datum["label"]

        if i % update_interval == 0 and i > 0:
            # Get network predictions.
            all_activity_pred = all_activity(spike_record, assignments, n_classes)
            proportion_pred = proportion_weighting(spike_record, assignments, proportions, n_classes)

            # Compute network accuracy according to available classification strategies.
            accuracy["all"].append(100 * torch.sum(labels.long() == all_activity_pred).item() / update_interval)
            accuracy["proportion"].append(100 * torch.sum(labels.long() == proportion_pred).item() / update_interval)

            #print(f"\nAll activity accuracy: {accuracy['all'][-1]:.2f} (last), {np.mean(accuracy['all']):.2f} (average), {np.max(accuracy['all']):.2f} (best)")
            #print(f"Proportion weighting accuracy: {accuracy['proportion'][-1]:.2f} (last), {np.mean(accuracy['proportion']):.2f} (average), {np.max(accuracy['proportion']):.2f} (best)\n")

            # Assign labels to excitatory layer neurons.
            assignments, proportions, rates = assign_labels(spike_record, labels, n_classes, rates)
            if plot:
                display(Javascript(f"addData({i},{np.mean(accuracy['all']):.2f})"))

        # Add the current label to the list of labels for this update_interval
        labels[i % update_interval] = label[0]

        # Run the network on the input.
        choice = np.random.choice(int(n_neurons / n_classes), size=1, replace=False)
        clamp = {"Ae": per_class * label.long() + torch.Tensor(choice).long()}
        inputs = {"X": image.cuda().view(time, 1, 1, 28, 28)}
        network.run(inputs=inputs, time=time, clamp=clamp)

        # Add to spikes recording.
        spike_record[i % update_interval] = spikes["Ae"].get("s").view(time, n_neurons)

        network.reset_state_variables()  # Reset state variables.
        if not plot:
            pbar.set_description_str(f"Accuracy: {np.mean(accuracy['all']):.2f}. Train progress: ")
            pbar.update()

    print(f"Progress: {n_train} / {n_train} \n")
    print("Training complete.\n")
    return network

# do przerobienia
def test_network(network, config, spikes):
    print("Testing....\n")
    # Load MNIST data.
    test_dataset = MNIST(
        PoissonEncoder(time=config["time"], dt=config["dt"]),
        None,
        root=os.path.join("..", "..", "data", "MNIST"),
        download=True,
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(lambda x: x * config["intensity"])]
        ),
    )

    # Sequence of accuracy estimates.
    accuracy = {"all": 0, "proportion": 0}

    # Record spikes during the simulation.
    spike_record = torch.zeros(1, int(config["time"] / config["dt"]), config["n_neurons"], device=device)

    # Train the network.
    print("\nBegin testing\n")
    network.train(mode=False)

    assignments = -torch.ones_like(torch.Tensor(config["n_neurons"]), device=device)
    proportions = torch.zeros_like(torch.Tensor(config["n_neurons"], config["n_classes"]), device=device)
    rates = torch.zeros_like(torch.Tensor(config["n_neurons"], config["n_classes"]), device=device)

    pbar = tqdm(total=config["n_test"])
    for step, batch in enumerate(test_dataset):
        if step > config["n_test"]:
            break
        # Get next input sample.
        inputs = {"X": batch["encoded_image"].view(int(config["time"] / config["dt"]), 1, 1, 28, 28)}
        inputs = {k: v.cuda() for k, v in inputs.items()}

        # Run the network on the input.
        network.run(inputs=inputs, time=config["time"], input_time_dim=1)

        # Add to spikes recording.
        spike_record[0] = spikes["Ae"].get("s").squeeze()

        # Convert the array of labels into a tensor
        label_tensor = torch.tensor(batch["label"], device=device)

        # Get network predictions.
        all_activity_pred = all_activity(spikes=spike_record, assignments=assignments, n_labels=config["n_classes"])
        proportion_pred = proportion_weighting(
            spikes=spike_record,
            assignments=assignments,
            proportions=proportions,
            n_labels=config["n_classes"],
        )

        # Compute network accuracy according to available classification strategies.
        accuracy["all"] += float(torch.sum(label_tensor.long() == all_activity_pred).item())
        accuracy["proportion"] += float(torch.sum(label_tensor.long() == proportion_pred).item())

        network.reset_state_variables()  # Reset state variables.

        pbar.set_description_str(f"Accuracy: {(max(accuracy['all'] ,accuracy['proportion'] ) / (step+1)):.3}")
        pbar.update()

    print(f"\nAll activity accuracy: {(accuracy['all'] / config['n_test']):.2f}")
    print(f"Proportion weighting accuracy: {(accuracy['proportion'] / config['n_test']):.2f} \n")

    print("Testing complete.\n")
    return network

def save_network(network, filename):
    if not os.path.exists("./models"):
        os.mkdir("./models")
    torch.save(network.state_dict(), os.path.join("./models", filename))
