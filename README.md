# HondaAI
UT Austin Senior Design Project 2025-2026
by Julian Rodriguez, Alex Johnson, Kellen Watts, Nathan Nguyen, Bryan Calixto, and Jay Kannam
# Quick Guide

## Setup
- Use Python 3.12.0
- Create a virtual environment within your IDE
- run ```pip install requirements.txt``` in your terminal. (May require some debugging depending on your system)

## Start
To run our system on a new participant (or just to try out the system) run the following:
- If you want to use the adaptive alert system: ```python alert_pipeline.py --calibration-duration 300 --participant-number X --session-duration 20 --av-speedup 2.5 --alert-mode adaptive --tutorial-duration 10 --headless-av```
- If you want to use the fixed alert system: ```python alert_pipeline.py --calibration-duration 300 --participant-number X --session-duration 20 --av-speedup 2.5 --alert-mode fixed --tutorial-duration 10 --headless-av```

Be sure to replace "X" with your participant's participant number. After each trial, a .csv file containing the logged data and .png file with a graph of the participant's scores will automatically be created.

### IMPORTANT: The participant number is what we use to identify a participant's specific trial. This means that if you want to save the data of a participant under the fixed alert system and adaptive alert system, you need to replace the participant number for each trial. For example, if you had a new participant in the simulator, you may assign him the participant number "0" for when he runs the adaptive alert system, and "1" for when he runs the fixed alert system.
