

# python compare_pose.py --reference tpose.json --frames frames.txt


#file compares a pose against a sequence of time frames (currently this is for a whole sequence against one pose but
#much of the functionality of comparing the correct sequence of poses to the whole time frames should be the
#the same function calls)

#this file takes in a pose from a json and a txt file for comparsion

#to use this  python3 comparsion_pose.py --reference tpose.json --frames pose_locations.txt

import json
import math
import re
import argparse



#the next three functions parases our data 

#converts our time frame samples to a regex (use whats generated from posenet.py)
LINE_PATTERN = re.compile(
    r"nose\s*->\s*(\w+)\s*:\s*\(\s*([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\s*\)"
)
TIMESTAMP_PATTERN = re.compile(r"Timestamp:\s*([\d.]+)s")

#loads a pose from a json file, please use the jsons from extract_pose.py
def load_reference(filepath: str) -> dict:
    with open(filepath) as file:
        data = json.load(file)
    return data  # { pose_name, keypoints: { name: {dx,dy,magnitude,angle_rad,...} } }

#same logic from extract_pose.txt except now works for a while txt file which contains a representation of our live timeframes
#this returns list of dicts whcih is just a time stap into a keypoints and the keypoints dx,dy and angle relative to the nose
def parse_frames(filepath: str) -> list[dict]:
    frames = []
    current_timestamp = None
    current_keypoints = {}

    with open(filepath) as file:
        for line in file:
            ts_match = TIMESTAMP_PATTERN.search(line)
            if ts_match:
                #saves timeframe if we have one
                if current_timestamp  is not None and current_keypoints:
                    frames.append({"timestamp": current_timestamp , "keypoints": current_keypoints})
                current_timestamp  = float(ts_match.group(1))
                current_keypoints = {}
                continue

            kp_match = LINE_PATTERN.search(line)
            if kp_match:
                name = kp_match.group(1)
                dx   = float(kp_match.group(2))
                dy   = float(kp_match.group(3))
                magnitude  = math.sqrt(dx * dx + dy * dy)
                angle_rad = math.atan2(dy, dx)
                current_keypoints[name] = {
                    "dx":dx,
                    "dy":dy,
                    "magnitude": round(magnitude,6),
                    "angle_rad": round(angle_rad,6),
                }

    # Don't forget the last frame
    if current_timestamp  is not None and current_keypoints:
        frames.append({"timestamp": current_timestamp , "keypoints": current_keypoints})

    return frames


#manually set joint weights for comparsion calculations later 
#this lets us set weights for what we actually want to include 1 incude 0 dont include
JOINT_WEIGHTS = {
    "left_shoulder":  1,
    "right_shoulder": 1,
    "left_elbow":     1,
    "right_elbow":    1,
    "left_wrist":     1,
    "right_wrist":    1,
    "left_hip":       1,
    "right_hip":      1,
    "left_knee":      1,
    "right_knee":     1,
    "left_ankle":     1,
    "right_ankle":    1,
    "neck":           1,
    "left_eye":       0,
    "right_eye":      0,
    "left_ear":       0,
    "right_ear":      0,
}


#below is the main comparsion logic used for this

#smallest angle difference between two angles [0,pi] or [0,180]
def _angle_diff(a: float, b: float) -> float:
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


#converts the angle to a score
#there is a exponential fall off e^(-curve * (error/pi)^2)
def _angle_to_score(angle_error_radians: float) -> float:
    
    #higher the curve, the less forgiving the angle error is
    curve: float = 4.0


    t = angle_error_radians / math.pi   #normalise to [0, 1] [0,180]
    return math.exp(-curve * t * t) * 100

#actual comparsion of a pose to a frame (start here)
def compare(reference: dict, frame: dict) -> dict:

    #loads pose reference keypoints
    reference_keypoints = reference["keypoints"]

    #loads the frame keypoint
    frame_keypoints = frame["keypoints"]

    #score
    joint_scores = {}

    #compares all joints in the reference pose to the frame and calculates an angle error and a score for each joint's
    #here the joint name is the keypoint
    for name, reference_keypoint in reference_keypoints.items():

        #if the joints dont exist we will just skip it as we can't compare it, idealy we always have all 
        #keypoints
        if name not in frame_keypoints:
            continue
        

        frame_keypoint  = frame_keypoints[name]
        angle_error = _angle_diff(reference_keypoint["angle_rad"], frame_keypoint["angle_rad"]) #calculates angle diff of the current keypoint for the timestamp and reference pose
        score     = _angle_to_score(angle_error) #convers angle diff to score
        weight    = JOINT_WEIGHTS.get(name, 1.0) #applies weight per joint keypoint as we will not consider some

        #stores score for that joint keypoint differnce of frame to pose reference
        #does it per joint keypoint
        joint_scores[name] = {
            "score":           round(score, 1),
            "angle_error_deg": round(math.degrees(angle_error), 2),
            "weight":          weight,
        }

    
    #here we only include the joints that we care about so its any of the joint that have weight set higher to our threshold
    heavy_scores = {
        k: v["score"] for k, v in joint_scores.items()
        if JOINT_WEIGHTS.get(k, 0) >= 1
    }
    worst_joint_name  = min(heavy_scores, key=heavy_scores.get) if heavy_scores else None
    worst_joint_score = heavy_scores[worst_joint_name] if worst_joint_name else 0.0

    return {
        "overall_score":    round(worst_joint_score, 2),
        "worst_joint_name": worst_joint_name,
        "joints":           joint_scores,
    }


# For visual output to terminal, this will be scraped later, generated via ai for visual display through testing

def print_frame_result(timestamp: float, result: dict):
    print(f"\n  Timestamp: {timestamp}s  |  Score: {result['overall_score']:.1f}%"
          f"  (worst limb: {result['worst_joint_name']})")
    print(f"  {'Joint':<18} {'Score':>7}  {'Angle Error':>12}  {'Weight':>6}")
    print(f"  {'-'*50}")
    for name, stats in result["joints"].items():
        bar     = "█" * int(stats["score"] / 10)
        marker  = " ◄" if name == result["worst_joint_name"] else ""
        print(f"  {name:<18} {stats['score']:>6.1f}%  {stats['angle_error_deg']:>10.2f}°  {stats['weight']:>6.1f}  {bar}{marker}")


# Pipeline done with the help of ai for testing of calculation and access of files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare poses against a reference.")
    parser.add_argument("--reference", required=True, help="Path to reference pose JSON")
    parser.add_argument("--frames",    required=True, help="Path to timestamped frames TXT")
    args = parser.parse_args()

    reference = load_reference(args.reference)
    frames    = parse_frames(args.frames)

    print(f"\nReference pose : {reference['pose_name']}")
    print(f"Frames file    : {args.frames}  ({len(frames)} frame(s) found)")
    print("=" * 60)

    results = []
    
    #this will do it for every time frame but compares one frame at a time
    for frame in frames:
        result = compare(reference, frame)
        results.append((frame["timestamp"], result))
        print_frame_result(frame["timestamp"], result)

    if results:
        best_ts,  best_result  = max(results, key=lambda x: x[1]["overall_score"])
        worst_ts, worst_result = min(results, key=lambda x: x[1]["overall_score"])
        avg_score = sum(r["overall_score"] for _, r in results) / len(results)

        print(f"\n{'='*60}")
        print(f"  Best  : {best_ts}s  →  {best_result['overall_score']:.1f}%  (worst limb: {best_result['worst_joint_name']})")
        print(f"  Worst : {worst_ts}s  →  {worst_result['overall_score']:.1f}%  (worst limb: {worst_result['worst_joint_name']})")
        print(f"  {'-'*56}")
        print(f"  Avg worst-limb score : {avg_score:.1f}%  across {len(results)} frame(s)")
        print(f"{'='*60}\n")
