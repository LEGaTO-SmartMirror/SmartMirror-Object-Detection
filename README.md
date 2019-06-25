# SmartMirror-Object-Detection

Uses the camera stream of the [SmartMirror-Camera-Publisher](https://github.com/LEGaTO-SmartMirror/SmartMirror-Camera-Publisher), which creates a appsink under /tmp/camera_image .
It returns a json object with the position, tracking id and name of each detected Object to the scope "DETECTED_OBJECTS".

An example for the json object can be seen below:

{"DETECTED_OBJECTS": [{"TrackID": 1.0, "center": [0.12593, 0.7125], "name": "chair", "w_h": [0.2463, 0.15417]}, {"TrackID": 3.0, "center": [0.18889, 0.79167], "name": "bottle", "w_h": [0.10741, 0.1724]}]}


## ACKNOWLEDGEMENT

This work has been supported by EU H2020 ICT project LEGaTO, contract #780681.
