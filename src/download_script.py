import airfrans as af
from pathlib import Path

def main():
    ROOT_DIR = Path(__file__).parent.parent
    DATA_PATH = ROOT_DIR/"data"

    # Check if the data already exists
    if(DATA_PATH.exists):
        af.dataset.download(root = DATA_PATH, file_name='Dataset', unzip = True, OpenFOAM = False)



if __name__ == "__main__":
    main()