import os
import random
from typing import Optional
from pathlib import Path
from plinder.core.scores import query_index
import yaml


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
    config['system_id'] = system_id
    with open(os.path.join(output_dir, f'{system_id}.yaml'), 'w') as f:
        yaml.dump(config, f)
    return os.path.join(output_dir, f'{system_id}.yaml')