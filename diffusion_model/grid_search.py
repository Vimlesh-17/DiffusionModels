import itertools
import torch
from diffusion_model.train import train
import plotly.graph_objects as go
import os

def grid_search(args, train_loader):
    learning_rates = [1e-4, 1e-3]
    batch_sizes = [32]
    noise_steps = [500, 1000]
    beta_starts = [1e-3, 5e-4]
    beta_ends = [0.02, 0.1]

    grid = list(itertools.product(learning_rates, batch_sizes, noise_steps, beta_starts, beta_ends))
    results = {}
    best_run = None
    best_loss = float('inf')

    for i, (lr, bs, ns, beta_start, beta_end) in enumerate(grid):
        args.lr = lr
        args.batch_size = bs
        args.noise_steps = ns
        args.beta_start = beta_start
        args.beta_end = beta_end
        args.run_name = f"run_{i}_lr_{lr}_bs_{bs}_ns_{ns}_beta_{beta_start}_to_{beta_end}"

        # Pass DataLoader (train_loader) directly to the train function
        dataset = train_loader.dataset
        loss_history = train(args, dataset)
        results[args.run_name] = loss_history

        # Check if this is the best run
        if min(loss_history) < best_loss:
            best_loss = min(loss_history)
            best_run = args.run_name

    return results, best_run


def save_results_as_html(results, best_run, html_filename="grid_search_results.html"):
    fig = go.Figure()
    for run_name, losses in results.items():
        fig.add_trace(go.Scatter(y=losses, mode='lines+markers', name=run_name))

    fig.update_layout(title="Grid Search Results", xaxis_title="Epoch", yaxis_title="MSE Loss")
    fig.write_html(html_filename)

    print(f"Best run: {best_run} with loss: {min(results[best_run])}")
