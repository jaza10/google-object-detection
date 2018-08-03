# google-object-detection
Python notebooks for Google Object Detection Challenge v4

To run notebook, please install ImageAI with its dependencies:
https://github.com/OlafenwaMoses/ImageAI

Especially, download the RetinaNet weights and put the weights in the same folder as the SimpleDetection notebook:
https://github.com/OlafenwaMoses/ImageAI/tree/master/imageai/Detection

The notebook accepts an image as an input and outputs the image with the predicted bounding boxes.
The notebook also outputs the prediction string for each detected object and bounding box.

![picture]./images/output/0a0a615629231821new.jpg

To Dos:
- Match image detections to Google classes
- Iterate over all existing test images
- Submit first submission file to get status of prediction accuracy
