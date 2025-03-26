import os
from functools import partial
import pickle

import datasets
import timm

from torch import optim

from SHAP_MIA.model.federated_model import FederatedModel
from SHAP_MIA.node.federated_node import FederatedNode
from SHAP_MIA.simulation.simulation import Simulation
from SHAP_MIA.aggregators.fedopt_aggregator import Fedopt_Optimizer
from SHAP_MIA.files.archive import create_archive
from opacus.validators import ModuleValidator
from opacus import PrivacyEngine

DATASET_PATH = r'/home/mzuziak/archives/MIA_SHAP/experiments/datasets/uniform/cifar10-5/CIFAR10_5_dataset'
NET_ARCHITECTURE = timm.create_model('resnet34', num_classes=10, pretrained=False, in_chans=3)
NUMBER_OF_CLIENTS = 5
ITERATIONS = 50
LOCAL_EPOCHS = 3
LOADER_BATCH_SIZE = 16
LEARNING_RATE = 0.001
ARCHIVE_PATH = os.getcwd()
ARCHIVE_NAME = 'withoutDP_uniform_cifar10-5'


def integration_test():
    (metrics_savepath, 
     nodes_models_savepath, 
     orchestrator_model_savepath) = create_archive(
         path=ARCHIVE_PATH,
         archive_name=ARCHIVE_NAME
     )
    
    data = datasets.load_from_disk(DATASET_PATH)
    orchestrators_data = data['orchestrator_test_set']
    net_architecture = NET_ARCHITECTURE
    optimizer_architecture = partial(optim.SGD, lr=LEARNING_RATE)
    ##########################

    net_architecture = ModuleValidator.fix(net_architecture)
    dp_settings = {
        0: {
            'DP': False,
            'Privacy_Engine': None
        },
        1: {
            'DP': False,
            'Privacy_Engine': None
        },
        2: {
            'DP': False,
            'Privacy_Engine': None
        },
        3: {
            'DP': False,
            'Privacy_Engine': None
        },
        4: {
            'DP': False,
            'Privacy_Engine': None
        }
    }

    ##########################

    model_tempate = FederatedModel(
        net=net_architecture,
        optimizer_template=optimizer_architecture,
        loader_batch_size=LOADER_BATCH_SIZE
    )
    node_template = FederatedNode()
    fed_avg_aggregator = Fedopt_Optimizer()
    
    simulation_instace = Simulation(model_template=model_tempate,
                                    node_template=node_template)
    simulation_instace.attach_orchestrator_model(orchestrator_data=orchestrators_data)
    simulation_instace.attach_node_model({
            node: [data[f"client_{node}_train_set"], data[f"client_{node}_test_set"]] for node in range(NUMBER_OF_CLIENTS)},
            dp_settings = dp_settings
        )
    simulation_instace.training_protocol(
        iterations=ITERATIONS,
        sample_size=NUMBER_OF_CLIENTS,
        local_epochs=LOCAL_EPOCHS,
        aggrgator=fed_avg_aggregator,
        shapley_processing_batch=NUMBER_OF_CLIENTS,
        metrics_savepath=metrics_savepath,
        nodes_models_savepath=nodes_models_savepath,
        orchestrator_models_savepath=orchestrator_model_savepath
    )


if __name__ == "__main__":
    integration_test()