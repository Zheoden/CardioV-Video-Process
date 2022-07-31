Video and frame processing for CardioV Software

Functionality
Video and Image border detection using OpenCV.

Instructions
1. Execute python file edge_detector.py providing 2 arguments:
  a. 'v' or 'i' to specify the type of file to process (video or image)
  b. Path to de file
2. A processed image will display. In case of a video, as soon as you close one image another one will pop up until the video is out of frames

Next Steps:
1. Crop the images to show only the heart area to make it easier to measure its chambers
2. Get chambers measure
3. Bug fixing

Known bugs:
1. When displaying video frames, script execution can get stuck if a key is being pressed continuously and then closing all frame windows at once (pressing a key causes another frame to display)
