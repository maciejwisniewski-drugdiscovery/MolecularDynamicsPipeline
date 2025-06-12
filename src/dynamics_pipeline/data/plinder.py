import os
import random
from typing import Optional
from pathlib import Path
from plinder.core.scores import query_index
import yaml
from dotenv import load_dotenv
load_dotenv()

class PlinderFilters:
    def __init__(self, filters: Path):
        
        config = self.read_yaml(filters)
        
        self.columns = config["columns"]
        self.splits = config["splits"]
        self.filters = [(filter['column'], filter['operator'], filter['value']) for filter in  config["filters"]]

    def read_yaml(self, filters: Path):
        with open(filters, "r") as f:
            config = yaml.safe_load(f)
        return config



def load_plinder_ids(filters: Optional[str] = None):
    plinder_filters = PlinderFilters(filters)
    PLINDEX = query_index(
        columns = plinder_filters.columns, 
        splits = plinder_filters.splits,
        filters = plinder_filters.filters
        )
    plinder_ids = set(PLINDEX['system_id'].tolist())
    return plinder_ids

def create_system_config(template_config: Path, system_id: str, output_dir: Path):
    with open(template_config, 'r') as f:
        config = yaml.safe_load(f)

    config['info']['system_id'] = system_id
    if config['info']['bound_state'] == True:
        config['info']['simulation_id'] = f"{system_id}_simulation_bound_state"
    else:
        config['info']['simulation_id'] = f"{system_id}_simulation_unbound_state"

    system_config_filepath = os.path.join(output_dir, config['info']['simulation_id'], 'config.yaml')

    if not os.path.exists(system_config_filepath):
        config['paths']['raw_protein_files'] = [os.getenv('PLINDER_MOUNT'),os.getenv('PLINDER_RELEASE'),os.getenv('PLINDER_ITERATION'),'systems',system_id,'protein.pdb']
        config['paths']['raw_ligand_files'] = [os.getenv('PLINDER_MOUNT'),os.getenv('PLINDER_RELEASE'),os.getenv('PLINDER_ITERATION'),'systems',system_id,'ligand.sdf']
        config['paths']['output_dir'] = os.path.join(output_dir, config['info']['simulation_id'])
        os.makedirs(os.path.join(output_dir, config['info']['simulation_id']), exist_ok=True)
        with open(system_config_filepath, 'w') as f:
            yaml.dump(config, f)

    return system_config_filepath