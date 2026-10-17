import airfrans as af
from pathlib import Path
from zipfile import ZipFile

def main():
    ROOT_DIR = Path(__file__).resolve().parent.parent
    DATA_PATH = ROOT_DIR/"data"
    ZIP_PATH = DATA_PATH/"Dataset.zip"
    DATASET_PATH = DATA_PATH/"Dataset"

    # Check if the data already exists
    if DATASET_PATH.exists():
        print("Already prepared")
        return
    elif not ZIP_PATH.exists():
        print("ZIP FIle missing please run:")
        print("curl.exe -L -C - -o data\Dataset.zip https://data.isir.upmc.fr/extrality/NeurIPS_2022/Dataset.zip")
        return
    else:
        # Test that the extracted file has no problems
        with ZipFile(ZIP_PATH,"r") as zObject:
            failures = zObject.testzip()

        if failures :
            print("Failures detected in:")
            print(failures)
        else:
            with ZipFile(ZIP_PATH,"r") as zObject:
                zObject.extractall(DATA_PATH)
            print("done")
        
if __name__ == "__main__":
    main()