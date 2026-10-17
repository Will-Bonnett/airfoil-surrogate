from pathlib import Path
from parse_name import parse_name
import airfrans as af
import numpy as np
import json
from tqdm import tqdm

def main():
    DATA_PATH = ROOT_DIR = Path(__file__).resolve().parent.parent/"data"/"Dataset"
    
    # Load JSON
    with open(DATA_PATH/"manifest.json",'r') as json_file:
        manifest = json.load(json_file)
    data_set = set(manifest["full_train"]).union(manifest["full_test"])
    
    if len(data_set) != 1000: raise ValueError("Dataset doesn't contain 1000 records")

    failures = []
    for name in tqdm(data_set):
        try:
            dict = parse_name(name)
            simulation = af.Simulation(DATA_PATH,name=name)
            if not np.isfinite(simulation.velocity).all():
                raise ValueError("Simulation contains infinite value(s)")
            if not np.isfinite(simulation.pressure).all():
                raise ValueError("Simulation contains infinite value(s)")

        except Exception as e:
            failures.append((name,repr(e)))

    print(f"{1000-len(failures)}/1000 passed")
    if failures:
        for name, err in failures[:10]:
            print(f" FAIL {name}: {err}")
        sys.exit(1)

    

        
    
if __name__ == "__main__":
    main()