# Hardware dataset format v1

A single data collection run involves:
1) Placing a piece of paper on the printer surface.
2) Running the data collection script. The printer will zero and allow a pen to be inserted.
3) Type the name of the pen and hit enter to start collection.
4) Eventually control-C to stop collection.

The data directory will contain:
- A `info.json` JSON file containing:
    - The dataset format version.
    - The supplied unique string name of the pen type.
    - A dictionary of action generation parameters.
    - A list of the actions taken, as a dictionary of:
       - Action representation: unique action type plus action-type-dependent fields.
       - Before image: image name
       - After image: image name
- The set of referenced images. These are rectified, white-balanced images of the canvas taken between paintstrokes.