from hashlib import file_digest
from pathlib import Path
from shutil import copyfileobj
from urllib.request import urlopen


DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
DATASET_SHA256 = "a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20"


def has_expected_checksum(path):
    with path.open("rb") as source:
        return file_digest(source, "sha256").hexdigest() == DATASET_SHA256


def main():
    dataset = Path("data/ml-1m.zip")
    if dataset.is_file() and has_expected_checksum(dataset):
        print(f"Using {dataset}")
        return

    dataset.parent.mkdir(parents=True, exist_ok=True)
    partial = dataset.with_suffix(".part")
    try:
        with urlopen(DATASET_URL, timeout=90) as response, partial.open("wb") as output:
            copyfileobj(response, output)
        if not has_expected_checksum(partial):
            raise ValueError("MovieLens archive checksum mismatch")
        partial.replace(dataset)
    finally:
        partial.unlink(missing_ok=True)
    print(f"Downloaded {dataset}")


if __name__ == "__main__":
    main()
