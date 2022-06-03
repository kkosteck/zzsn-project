import os, json

import numpy as np
import torch
from tqdm import tqdm

from bindsnet.evaluation import assign_labels, proportion_weighting
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.manual_seed_all(0)
torch.set_num_threads(os.cpu_count() - 1)


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
        reduction = torch.sum,
    )
    network.to("cuda")

    return network

def create_monitorings(network: DiehlAndCook2015, time: int):
    spikes = {}
    for layer in set(network.layers):
        spikes[layer] = Monitor(network.layers[layer], state_vars=["s"], time=time)
        network.add_monitor(spikes[layer], name=f"{layer}_spikes")

    return network, spikes

def train_network(network, train_set, val_set, spikes, config):
    dataloader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)
    spike_record = torch.zeros(config["update_interval"], config["time"], config["n_neurons"], device=device)
    accuracy = []
    labels = torch.empty(config["update_interval"], device=device)
    per_class = int(config["n_neurons"] / config["n_classes"])

    assignments = -torch.ones_like(torch.Tensor(config["n_neurons"]), device=device)
    proportions = torch.zeros_like(torch.Tensor(config["n_neurons"], config["n_classes"]), device=device)
    rates = torch.zeros_like(torch.Tensor(config["n_neurons"], config["n_classes"]), device=device)
    # Train the network.
    print("Begin training.\n")

    network.train()
    pbar = tqdm(total=config["n_train"], position=0, leave=True)
    for (i, datum) in enumerate(dataloader):
        if i > config["n_train"]:
            break

        image = datum["encoded_image"]
        label = datum["label"]

        if i % config["update_interval"] == 0 and i > 0:
            # Get network predictions.
            proportion_pred = proportion_weighting(spike_record, assignments, proportions, config["n_classes"])

            # Compute network accuracy according to available classification strategies.
            accuracy.append(100 * torch.sum(labels.long() == proportion_pred).item() / config["update_interval"])
            with open("acc_train.json", "w") as f:
                json.dump(accuracy, f, indent=4)

            # Assign labels to excitatory layer neurons.
            assignments, proportions, rates = assign_labels(spike_record, labels, config["n_classes"], rates)
        if i% config["eval_interval"] == 0 and i > 0:
            acc = validate_network(network, val_set, spikes, config)
            print(f"\nValidation accuracy: {acc:.2f}")

        # Add the current label to the list of labels for this update_interval
        labels[i % config["update_interval"]] = label[0]

        # Run the network on the input.
        choice = np.random.choice(int(config["n_neurons"] / config["n_classes"]), size=1, replace=False)
        clamp = {"Ae": per_class * label.long() + torch.Tensor(choice).long()}
        inputs = {"X": image.cuda().view(config["time"], 1, 1, 28, 28)}
        network.run(inputs=inputs, time=config["time"], clamp=clamp)

        # Add to spikes recording.
        spike_record[i % config["update_interval"]] = spikes["Ae"].get("s").view(config["time"], config["n_neurons"])

        network.reset_state_variables()  # Reset state variables.
        pbar.set_description_str(f"Accuracy: {np.mean(accuracy):.2f}. Train progress: ")
        pbar.update()

    print(f"Progress: {config['n_train']} / {config['n_train']} \n")
    print("Training complete.\n")
    return network

def validate_network(network, val_set, spikes, config):
    print("Evaluate network...")
    network.eval()
    accuracy = 0
    with torch.no_grad():

        spike_record = torch.zeros(1, int(config["time"] / config["dt"]), config["n_neurons"], device=device)
        assignments = -torch.ones_like(torch.Tensor(config["n_neurons"]), device=device)
        proportions = torch.zeros_like(torch.Tensor(config["n_neurons"], config["n_classes"]), device=device)

        pbar = tqdm(total=config["n_eval"], position=0, leave=True)
        for step, batch in enumerate(val_set):
            if step > config["n_eval"]:
                break
            inputs = {"X": batch["encoded_image"].view(int(config["time"] / config["dt"]), 1, 1, 28, 28)}
            inputs = {k: v.cuda() for k, v in inputs.items()}

            network.run(inputs=inputs, time=config["time"], input_time_dim=1)
            spike_record[0] = spikes["Ae"].get("s").squeeze()
            label_tensor = torch.tensor(batch["label"], device=device)

            proportion_pred = proportion_weighting(
                spikes=spike_record,
                assignments=assignments,
                proportions=proportions,
                n_labels=config["n_classes"],
            )

            accuracy += float(torch.sum(label_tensor.long() == proportion_pred).item())

            network.reset_state_variables() 

            pbar.set_description_str(f"Accuracy: {(accuracy / (step+1)):.3}")
            pbar.update()
        
    network.train()
    return 100 * accuracy/ config['n_eval']
    

def test_network(network, config, spikes, test_dataset):
    print("Testing....\n")
    accuracy = []

    # Record spikes during the simulation.
    print(config["time"])
    spike_record = torch.zeros(1, int(config["time"] / config["dt"]), config["n_neurons"], device=device)

    print("\nBegin testing\n")
    network.train(mode=False)

    assignments = -torch.ones_like(torch.Tensor(config["n_neurons"]), device=device)
    proportions = torch.zeros_like(torch.Tensor(config["n_neurons"], config["n_classes"]), device=device)
    rates = torch.zeros_like(torch.Tensor(config["n_neurons"], config["n_classes"]), device=device)

    pbar = tqdm(total=config["n_test"], position=0, leave=True)
    for step, batch in enumerate(test_dataset):
        if step > config["n_test"]:
            break
        # Get next input sample.
        inputs = {"X": batch["encoded_image"].view(int(config["time"] / config["dt"]), 1, 1, 28, 28)}
        inputs = {k: v.cuda() for k, v in inputs.items()}

        # Run the network on the input.
        network.run(inputs=inputs, time=config["time"], input_time_dim=1)

        spike_record[0] = spikes["Ae"].get("s").squeeze()
        label_tensor = torch.tensor(batch["label"], device=device)
        proportion_pred = proportion_weighting(
            spikes=spike_record,
            assignments=assignments,
            proportions=proportions,
            n_labels=config["n_classes"],
        )

        accuracy.append(float(torch.sum(label_tensor.long() == proportion_pred).item()))
        with open("acc_test.json", "w") as f:
            json.dump(accuracy, f, indent=4)

        network.reset_state_variables()  # Reset state variables.

        pbar.update()

    print("Testing complete.\n")
    return network

def save_network(network, filename):
    if not os.path.exists("./models"):
        os.mkdir("./models")
    torch.save(network.state_dict(), os.path.join("./models", filename))
