import os
import yaml
import argparse
from dynamics_pipeline.data.misato import load_misato_ids, generate_misato_unbound_data
import logging
from multiprocessing import Pool, cpu_count


logger = logging.getLogger("misato_test")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def process_misato_id(args_dict):
    try:
        misato_id = args_dict['misato_id']
        generate_misato_unbound_data(misato_id,
                                     ccd_pkl_filepath = args_dict['ccd_pkl'],
                                     misato_dir = args_dict['misato_dir'],
                                     output_dir = args_dict['output_dir'],
                                     method = args_dict['method'],
                                     offset_a = args_dict['offset_a'],
                                     distance_a = args_dict['distance_a'],
                                     distance_b = args_dict['distance_b'],
                                     distance_c = args_dict['distance_c'],
                                     num_conformers_to_generate = args_dict['num_conformers_to_generate'],
                                     n_samples = args_dict['n_samples'],
                                     z_samples = args_dict['z_samples'],
                                     max_trials = args_dict['max_trials'],
                                     logger=logger)
    except Exception as e:
        logger.error(f"Error processing misato ID {args_dict['misato_id']}: {e}")
        return

def main():
    parser = argparse.ArgumentParser(description='Run Molecular Dynamics Simulation with OpenMM')
    parser.add_argument('--output_dir', type=str, required=False, help='Output directory', default=None)
    parser.add_argument('--misato_ids_filepath', type=str, required=False, help='Path to misato ids file', default=None)
    parser.add_argument('--misato_dir', type=str, required=False, help='Path to misato dir', default=None)
    parser.add_argument('--ccd_pkl', type=str, required=False, help='Path to ccd pkl file', default=None)
    parser.add_argument('--method', type=str, required=False, help='Method', default='mcmc')
    parser.add_argument('--offset_a', type=float, required=False, help='Offset a', default=8)
    parser.add_argument('--distance_a', type=float, required=False, help='Maximum distance between ligand to protein', default=24)
    parser.add_argument('--distance_b', type=float, required=False, help='Minimum distance between ligand conformers', default=0.5)
    parser.add_argument('--distance_c', type=float, required=False, help='Minimum distance between protein and ligand conformers', default=6)
    parser.add_argument('--num_conformers_to_generate', type=int, required=False, help='Number of conformers to generate', default=60)
    parser.add_argument('--n_samples', type=int, required=False, help='Number of samples', default=10000)
    parser.add_argument('--z_samples', type=int, required=False, help='Number of z samples', default=500)
    parser.add_argument('--max_trials', type=int, required=False, help='Maximum trials', default=10000000)
    parser.add_argument('--num_cpus', type=int, required=False, help='Number of CPUs to use for parallel processing (default: 1)', default=1)

    args = parser.parse_args()
    print(f"Loading misato ids from {args.misato_ids_filepath}")
    misato_ids = load_misato_ids(args.misato_ids_filepath)
    
    # Prepare arguments for parallel processing
    process_args = []
    for misato_id in misato_ids:
        if os.path.exists(os.path.join(args.output_dir, misato_id)):
            if len(os.listdir(os.path.join(args.output_dir, misato_id))) == args.z_samples:
                logger.info(f"Found {args.z_samples} conformations for {misato_id} in {os.path.join(args.output_dir, misato_id)}. Skipping generation.")
                continue
            else:
                logger.info(f"Found {len(os.listdir(os.path.join(args.output_dir, misato_id)))} conformations for {misato_id} in {os.path.join(args.output_dir, misato_id)}. Generating more conformations.")
        else:
            logger.info(f"No directory found for {misato_id} in {args.output_dir}. Creating directory.")
        args_dict = {
            'misato_id': misato_id,
            'ccd_pkl': args.ccd_pkl,
            'misato_dir': args.misato_dir,
            'output_dir': args.output_dir,
            'method': args.method,
            'offset_a': args.offset_a,
            'distance_a': args.distance_a,
            'distance_b': args.distance_b,
            'distance_c': args.distance_c,
            'num_conformers_to_generate': args.num_conformers_to_generate,
            'n_samples': args.n_samples,
            'z_samples': args.z_samples,
            'max_trials': args.max_trials
        }
        process_args.append(args_dict)

    # Use all available CPUs if num_cpus not specified
    num_cpus = args.num_cpus if args.num_cpus is not None else cpu_count()
    logger.info(f"Processing {len(misato_ids)} misato IDs using {num_cpus} CPUs")
    
    # Run parallel processing
    with Pool(num_cpus) as pool:
        pool.map(process_misato_id, process_args)

if __name__ == "__main__":
    main()


