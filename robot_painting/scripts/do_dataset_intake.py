import argparse
import ujson as json
import pathlib
import numpy as np
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Iterates through all data folders in the specified dataset directory, performing train/test/val assignment.")
    parser.add_argument("dataset_directory", type=str, help="Dataset directory to process")
    parser.add_argument("--reassign-all", action="store_true", help="Regenerate all dataset assignments. Note that this overwrites the old ones!")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_fraction", type=float, default=0.2)
    parser.add_argument("--val_fraction", type=float, default=0.0)

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    dataset_directory = pathlib.Path(args.dataset_directory)
    assert dataset_directory.exists()

    rng = np.random.default_rng(seed=args.seed)
    subdirectories = [x for x in dataset_directory.iterdir() if x.is_dir()]

    assert args.test_fraction >= 0 and args.val_fraction >= 0 and args.test_fraction + args.val_fraction < 1.
    def random_assignment():
        return rng.choice(["train", "test", "val"], p=[1. - args.test_fraction - args.val_fraction, args.test_fraction, args.val_fraction])

    total_assignments = {
        "train": 0,
        "test": 0,
        "val": 0
    }
    num_assigned = 0

    for subdirectory in subdirectories:
        info_json = subdirectory / "info.json"
        if not info_json.exists():
            continue
        print(f"Processing {info_json}.")
        with open(info_json, "r") as f:
            data = json.load(f)
        global_dataset_assignment = None
        if "dataset_assigment" in data:
            global_dataset_assignment = data["dataset_assignment"]
        
        for entry in data["actions"]:
            if args.reassign_all or "dataset_assignment" not in entry or entry["dataset_assignment"] == "unassigned":
                assignment = global_dataset_assignment or random_assignment()
                entry["dataset_assignment"] = assignment
                total_assignments[assignment] += 1
                num_assigned +=1
        
        with open(info_json, "w") as f:
            json.dump(data, f, indent=2)
        
    print(f"Done. Made {num_assigned} assignments.")
    total = sum(list(total_assignments.values()))
    if total > 0:
        for key, assignments in total_assignments.items():
            print(f"\t{key}: {assignments}/{total} = {assignments/total*100:0.1f}%")



    


