import os
import yaml
import argparse
from dynamics_pipeline.data.misato import load_misato_ids, generate_misato_unbound_data
import logging


logger = logging.getLogger("misato_test")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def main():
    parser = argparse.ArgumentParser(description='Run Molecular Dynamics Simulation with OpenMM')
    parser.add_argument('--output_dir', type=str, required=False, help='Output directory', default=None)
    parser.add_argument('--misato_ids_filepath', type=str, required=False, help='Path to misato ids file', default=None)
    parser.add_argument('--misato_dir', type=str, required=False, help='Path to misato dir', default=None)
    parser.add_argument('--ccd_pkl', type=str, required=False, help='Path to ccd pkl file', default=None)
    parser.add_argument('--offset_a', type=float, required=False, help='Offset a', default=8)
    parser.add_argument('--distance_a', type=float, required=False, help='Distance a', default=24)
    parser.add_argument('--distance_b', type=float, required=False, help='Distance b', default=5)
    parser.add_argument('--num_conformers_to_generate', type=int, required=False, help='Number of conformers to generate', default=50)
    parser.add_argument('--n_samples', type=int, required=False, help='Number of samples', default=500)
    parser.add_argument('--z_samples', type=int, required=False, help='Number of z samples', default=100)
    parser.add_argument('--max_trials', type=int, required=False, help='Maximum trials', default=100000)

    args = parser.parse_args()
    
    misato_ids = load_misato_ids(args.misato_ids_filepath)
    
    for misato_id in misato_ids:
        generate_misato_unbound_data(misato_id,
                                ccd_pkl = args.ccd_pkl,
                                misato_dir = args.misato_dir,
                                output_dir = args.output_dir,
                                offset_a = args.offset_a,
                                distance_a = args.distance_a,
                                distance_b = args.distance_b,
                                num_conformers_to_generate = args.num_conformers_to_generate,
                                n_samples = args.n_samples,
                                z_samples = args.z_samples,
                                max_trials = args.max_trials,
                                logger=logger)
        a=1
if __name__ == "__main__":
    main()