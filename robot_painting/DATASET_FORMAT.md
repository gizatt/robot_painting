# Hardware dataset format v1

## Raw logs

A single data collection run involves:
1) Placing a piece of paper on the printer surface.
2) Running the data collection script. The printer will zero and allow a pen to be inserted.
3) Type the name of the pen and hit enter to start collection.
4) Eventually control-C to stop collection.

The data directory will contain:
- A `info.json` JSON file containing information about the run. See below for the format.
    - The dataset format version.
    - The supplied unique string name of the pen type.
    - A dictionary of action generation parameters.
    - A list of the actions taken, as a dictionary of:
       - Action representation: unique action type plus action-type-dependent fields.
       - Before image: image name
       - After image: image name
       - dataset_assignment: "unassigned", "train", "test", "val", etc.
    - An optional global dataset_assignment, if all entries in the whole file should be forced to be train/test/val/etc.
- The set of referenced images. These are rectified, white-balanced images of the canvas taken between paintstrokes.

These logs should be saved into a `raw_data` directory.

## Datasets

Datasets are distinct folders (e.g. `dataset_20240823`) containing a set of log
folders (generated via the above runs). When the dataset intake script is run
(or re-run), each `info.json` is opened and any unassigned `dataset_assignments`
are assigned uniformly at random to the desired sets (or forced to be assigned
to the requested global dataset assignment if available).

The dataset folder should also contain a `calibration.npz` folder that's valid for all data in the dataset.
(TODO(gizatt) This could be dataset-run specific, maybe?)