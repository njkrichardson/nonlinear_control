import argparse 
from glob import glob 
from pathlib import Path 
from typing import List

import tqdm 

from utils import LOG_DIRECTORY, setup_logger, serialize, DATA_DIRECTORY

parser = argparse.ArgumentParser()

def main(args): 
    # logging 
    log = setup_logger(__name__) 
    log.info("starting results parser") 

    # get candidate directories 
    results_directories: List[Path] = glob((LOG_DIRECTORY / "car_control" / "*").as_posix()) 
    results: List[Path] = [] 

    for dir in tqdm.tqdm(results_directories): 
        results_paths: List[Path] = glob((Path(dir) / "*.pkl").as_posix())
        results.extend(results_paths)

    log.info(f"Found {len(results)} results")

#    serialize(results, (DATA_DIRECTORY / "all_results").as_posix())

if __name__=="__main__": 
    args = parser.parse_args()
    main(args) 
