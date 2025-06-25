def code_cell(source, readonly=False, editable=True,tags=[]):
    """Create a code cell with common metadata patterns"""
    if readonly:
        metadata = {
            "editable": False,
            "deletable": False, 
            "tags": tags
        }
    elif not editable:
        metadata = {
            "editable": False,
            "deletable": True,
            "tags": tags
        }
    else:
        metadata = {
            "editable": True,
            "deletable": True,
            "tags": tags
        }
    
    if isinstance(source, str):
        import textwrap
        source = textwrap.dedent(source).strip()
        source = [line + '\n' for line in source.split('\n')]
    else:
        source = [line + '\n' if not line.endswith('\n') else line for line in source]
    
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": metadata,
        "outputs": [],
        "source": source
    }

def markdown_cell(source):
    """Create a markdown cell"""
    if isinstance(source, str):
        import textwrap
        source = textwrap.dedent(source).strip()
        source = [line + '\n' for line in source.split('\n')]
    else:
        source = [line + '\n' if not line.endswith('\n') else line for line in source]
    
    return {
        "cell_type": "markdown", 
        "metadata": {},
        "source": source
    }

def notebook(cells):
    """Create a complete notebook"""
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }


fl_server = '''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import flwr as fl
from typing import List, Tuple
from flwr.common import Metrics
import pickle
import warnings
import logging
import argparse


logging.getLogger("flwr").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="flwr")

# Parse command line arguments
parser = argparse.ArgumentParser(description='Flower Federated Learning Server')
parser.add_argument('--NUM_ROUNDS', type=int, default=15, help='Number of federated learning rounds')
parser.add_argument('--MIN_CLIENTS', type=int, default=1, help='Minimum number of clients required')
parser.add_argument('--MODEL_NAME', type=str, default='my_model', help='Model name for saving')

args = parser.parse_args()

NUM_ROUNDS = args.NUM_ROUNDS
MIN_CLIENTS = args.MIN_CLIENTS
MODEL_NAME = args.MODEL_NAME

print(f"Server Configuration:")
print(f"  - NUM_ROUNDS: {NUM_ROUNDS}")
print(f"  - MIN_CLIENTS: {MIN_CLIENTS}")
print(f"  - MODEL_NAME: {MODEL_NAME}")
print()

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    if not metrics:
        return {}
    
    accuracies = [num_examples * m.get("accuracy", m.get("eval_accuracy", 0)) 
                  for num_examples, m in metrics 
                  if "accuracy" in m or "eval_accuracy" in m]
    examples = [num_examples for num_examples, m in metrics 
                if "accuracy" in m or "eval_accuracy" in m]
    
    if not accuracies:
        return {}
    
    return {"accuracy": sum(accuracies) / sum(examples)}

def weighted_average_fit(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate training metrics from fit() calls"""
    if not metrics:
        return {}
    
    train_accuracies = [num_examples * m["train_accuracy"] for num_examples, m in metrics if "train_accuracy" in m]
    train_losses = [num_examples * m["train_loss"] for num_examples, m in metrics if "train_loss" in m]
    examples = [num_examples for num_examples, m in metrics if "train_accuracy" in m]
    
    aggregated = {}
    if train_accuracies and examples:
        aggregated["train_accuracy"] = sum(train_accuracies) / sum(examples)
    if train_losses and examples:
        aggregated["train_loss"] = sum(train_losses) / sum(examples)
    
    return aggregated

class ServerStrategy(fl.server.strategy.FedAvg):
    def __init__(self, model_save_path='final_model.pkl', **kwargs):
        super().__init__(**kwargs)
        self.model_save_path = model_save_path
        self.final_parameters = None
    
    def aggregate_fit(self, server_round, results, failures):
        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        
        if results:
            print(f"=== Round {server_round} Training Results ===")
            
            for i, (client_proxy, fit_res) in enumerate(results):
                client_metrics = fit_res.metrics
                num_examples = fit_res.num_examples
                client_id = client_proxy.cid 
                
                print(f"Client {client_id}: {num_examples} samples")
                if "train_accuracy" in client_metrics:
                    print(f"  - Training accuracy: {client_metrics['train_accuracy']:.4f}")
                if "train_loss" in client_metrics:
                    print(f"  - Training loss: {client_metrics['train_loss']:.4f}")
        
        if server_round == NUM_ROUNDS and parameters is not None:
            self.final_parameters = parameters
            self._save_parameters(parameters)
        
        return parameters, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        """Collect evaluation metrics"""
        aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)
        
        if results:
            print(f"=== Round {server_round} Evaluation Results ===")
            for i, (client_proxy, eval_res) in enumerate(results):
                client_metrics = eval_res.metrics
                print(f"Client {i+1}: Loss={eval_res.loss:.4f}")
                if "eval_accuracy" in client_metrics:
                    print(f"  - Eval accuracy: {client_metrics['eval_accuracy']:.4f}")
                if "eval_loss" in client_metrics:
                    print(f"  - Eval loss: {client_metrics['eval_loss']:.4f}")
        
        return aggregated_metrics

    def _save_parameters(self, parameters):
        params_list = fl.common.parameters_to_ndarrays(parameters)
        
        model_data = {
            'parameters': params_list,
            'parameter_shapes': [param.shape for param in params_list],
            'parameter_dtypes': [str(param.dtype) for param in params_list],
            'num_parameters': len(params_list),
            'total_size': sum(param.size for param in params_list)
        }
        
        with open(self.model_save_path, 'wb') as f:
            pickle.dump(model_data, f,protocol=pickle.HIGHEST_PROTOCOL)
        

strategy = ServerStrategy(
    model_save_path='{}.pkl'.format(MODEL_NAME),
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=MIN_CLIENTS,
    min_evaluate_clients=MIN_CLIENTS,
    min_available_clients=MIN_CLIENTS, 
    evaluate_metrics_aggregation_fn=weighted_average,
    fit_metrics_aggregation_fn=weighted_average_fit  
)

def main():
    fl.server.start_server(
        server_address="0.0.0.0:9999",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
'''