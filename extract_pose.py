#this file will extract a pose based on a set of inputs that you need to manually input
#the input should be one fram from pose_location.txt.
#this will then manually make a pose and saving the displacement of each joint relatvie
#to the nose and will also calcualte the angle relative to that displacement

import json
import math
import re #regular expression libaray from python


#input raw text string here
RAW_POSE_STR = """
      neck -> nose: (0.3, 41.1)
      neck -> left_eye: (-10.7, 49.4)
      neck -> right_eye: (10.7, 49.4)
      neck -> left_ear: (-24.3, 45.2)
      neck -> right_ear: (24.3, 45.2)
      neck -> left_shoulder: (-46.7, 9.0)
      neck -> right_shoulder: (46.7, -9.0)
      neck -> left_elbow: (-127.0, 9.1)
      neck -> right_elbow: (80.6, -75.5)
      neck -> left_wrist: (-116.6, 59.3)
      neck -> right_wrist: (112.8, -130.1)
      neck -> left_hip: (-51.0, -129.1)
      neck -> right_hip: (18.6, -141.0)
      neck -> left_knee: (-115.1, -177.6)
      neck -> right_knee: (-1.7, -257.1)
      neck -> left_ankle: (-119.9, -293.0)
      neck -> right_ankle: (-16.3, -364.6)
"""

#set name of pose that the frame above should be and set output file name
POSE_NAME   = "left_arm_up_and_left_leg_up"
OUTPUT_FILE = "left_arm_up_and_left_leg_up.json"


#this is a python regex converter allowing us to parse through our RAW_POSE_STR
#this is soley for extraction and was done with the help of ai
LINE_PATTERN = re.compile(
    r"neck\s*"        # literal "neck", then optional whitespace
    r"->\s*"          # literal "->", then optional whitespace
    r"(\w+)\s*"       # capture group 1: destination keypoint name (letters, digits, underscores)
    r":\s*"           # literal ":", then optional whitespace
    r"\(\s*"          # literal "(", then optional whitespace
    r"([+-]?\d+\.?\d*)"  # capture group 2: dx — optional sign, digits, optional decimal
    r"\s*,\s*"        # literal ",", surrounded by optional whitespace
    r"([+-]?\d+\.?\d*)"  # capture group 3: dy — same number format as dx
    r"\s*\)"          # optional whitespace, then literal ")"
)

#this parses through our regex expression from above and extracts the displacement
#of the joint and its displacement relative to the nose
#This will then be stored in a dictionary called keypoints where the index is a name
#and that name holds a dx,dy which is the displacement relativeto the nose
def parse_vectors(raw: str) -> dict:
    keypoints = {}
    for line in raw.strip().splitlines():
        match = LINE_PATTERN.search(line)
        if not match:
            continue
        name = match.group(1) #extracts group one from above and saves it as a name
        dx   = float(match.group(2)) #extract group two from above and saves it as a float
        dy   = float(match.group(3)) #extract group three from above and saves it as a float
        keypoints[name] = (dx, dy) #saves this in dict keypoints
    return keypoints


#this is the main math for actually building a pose to where we will 
#take a dictionary (key to tuple data) here it will be our keypoints data(location of each
#joint relative to the nose and save it as a pose)
#here we will store the magnitude which is the length of the joint relative to nose
#we will also store the angle in radians using atan2 (we also store a degree version for readiablity)
def build_pose(keypoints: dict) -> dict:
    pose = {}

    for name, (dx, dy) in keypoints.items():

        #square root of (x^2 + y^2)
        magnitude = math.sqrt(dx * dx + dy * dy)

        #atan of y/x
        angle_rad = math.atan2(dy, dx)

        #store information in dict pose
        #here we round to 6 decimals as cutoff
        pose[name] = {
            "dx":dx,
            "dy":dy,
            "magnitude":round(magnitude, 6),
            "angle_rad":round(angle_rad, 6),
            "angle_deg":round(math.degrees(angle_rad), 6),
        }

    return pose


#if __name__ == "__main__": guards main functions if this needs to be inported later for helper functions
if __name__ == "__main__":

    #generates keypoints in our poses
    keypoints = parse_vectors(RAW_POSE_STR)

    #stores the magnitude and angle of each keypoint relative to the nose representing a pose
    pose = build_pose(keypoints)
 
    output = {
        "pose_name": POSE_NAME,
        "keypoints": pose,
    }

    with open(OUTPUT_FILE, "w") as file:
        json.dump(output, file, indent=2)
 
 
