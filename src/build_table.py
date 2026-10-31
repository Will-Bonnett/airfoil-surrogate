import json
import pandas as pd
import airfrans as af
from tqdm import tqdm
from pathlib import Path
from parse_name import parse_name
from physics import reynolds_number
from forces import pressure_force_coefficients

def process_data():
    """
    Build and Saves CSV of all 1000 simulations saving their
        - Name
        - The parsed Inputs
        - Re (Reynolds Number)
        - Targets from libary (including viscous part):
            - cl (coefficient of lift)
            - cd (coefficient of drag)
        - Library Parts:
            - cl_p (pressure)
            - cl_v (viscous)
            - cd_p (pressure)
            - cd_v (viscous)
        - Calculated cl_p and cd_p (forces.py)
        - Simulation Split
    """
    DATA_PATH = Path(__file__).resolve().parent.parent/"data"/"Dataset"
    CSV_PATH = Path(__file__).resolve().parent.parent/"data"/"processed"/"airfrans_coefficients.csv"

    COLUMNS = [
        # identity
        "name",
        # inputs
        "inletVelocity", "Re", "angleOfAttack",
        "M", "L", "P", "Q", "XX",
        # targets (library totals)
        "cl", "cd",
        # library parts
        "cl_p_lib", "cl_v_lib", "cd_p_lib", "cd_v_lib",
        # own cross-check
        "cl_p_cal", "cd_p_cal",
        # splits
        "split_full", "split_scarce", "split_reynolds", "split_aoa",
    ]

    # Load JSON
    with open(DATA_PATH/"manifest.json",'r') as json_file:
        manifest = json.load(json_file)
    full_train = set(manifest["full_train"])
    full_test = set(manifest["full_test"])
    aoa_train = set(manifest["aoa_train"])
    aoa_test = set(manifest["aoa_test"])
    scarce_train = set(manifest["scarce_train"])
    reynolds_train = set(manifest["reynolds_train"])
    reynolds_test = set(manifest["reynolds_test"])

    data = []
    # Iterate through each of those simulations and append the dictionary
    for name in tqdm(manifest["full_train"]+manifest["full_test"]):
        record = {"name": name} # Add name field
        inputs = parse_name(name) # parsed inputs
        re = reynolds_number(inputs["inletVelocity"]) # Calculate Reynolds Number

        simulation = af.Simulation(DATA_PATH, name)

        # Get Libary values
        (cd, cd_p_lib, cd_v_lib), (cl, cl_p_lib, cl_v_lib) = simulation.force_coefficient()
        cd_p_cal, cl_p_cal = pressure_force_coefficients(simulation)

        others =  {
            "Re":re,
            "cl":cl,
            "cd":cd,
            "cl_p_lib":cl_p_lib,
            "cl_v_lib":cl_v_lib,
            "cd_p_lib":cd_p_lib,
            "cd_v_lib":cd_v_lib,
            "cl_p_cal":cl_p_cal,
            "cd_p_cal":cd_p_cal,
            "split_full":checks_split(name,full_train, full_test),
            "split_scarce":checks_split(name, scarce_train, full_test),
            "split_reynolds":checks_split(name,reynolds_train,reynolds_test),
            "split_aoa":checks_split(name,aoa_train,aoa_test)}

        data.append(record | inputs | others)
    
    missing = set(pd.DataFrame(data).columns) - set(COLUMNS)
    assert not missing, f"columns not in COLUMNS: {missing}"


    # write to CSV
    df = pd.DataFrame(data)[COLUMNS]
    df.to_csv(CSV_PATH, index=False)

def checks_split(name, train, test):
    if name in train:
        return "train"
    if name in test:
        return "test"
    return None

if __name__ == "__main__":
    process_data()