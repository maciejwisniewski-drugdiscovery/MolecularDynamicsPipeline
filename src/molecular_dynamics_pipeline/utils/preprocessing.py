import os
import requests
import jinja2


import os
import requests
from tqdm import tqdm
from pathlib import Path
import dotenv
from biotite.structure.info import all_residues
from jinja2 import Template

# Load .env variables
dotenv.load_dotenv('poseidon.env')


# Define Jinja2 template
template = Template("https://files.rcsb.org/ligands/download/{{ residue }}_ideal.sdf")

def download_nonstandard_residue(residue: str, residue_filepath: Path):
    """
    Download the ideal structure of a given residue if not already present.
    """
    if residue_filepath.exists():
        return f"Skipped {residue}"

    url = template.render(residue=residue)
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(residue_filepath, "wb") as f:
            f.write(response.content)
        return True
    except Exception as e:
        print(f"Failed to download {residue}: {e}")
        return False
