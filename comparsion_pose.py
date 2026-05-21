import json
import math
import re


def parse_pose_sequence_data(filename: str) -> tuple[list[str], dict]:
    #opens a json file in read mode, this should be the song file sent from server
    with open(filename, "r") as f:
        data = json.load(f)

    #extracts the sequence order of poses from the json file, this is just 
    #a string of pose names in the order they should be performed in the song
    sequence_order = data["sequence_order"].split(", ")

    #extracts the poses from the json file, this is a list of dicts with each dict containing 
    #a pose name and the keypoints for that pose
    #poses here is the dictionary of pose names to their keypoints
    #for each pose in poses we going to store the pose name and its keypoints in a dictionary
    poses = {}
    for pose in data["poses"]:
        poses[pose["pose_name"]] = pose["keypoints"]

    return sequence_order, poses


def parse_frame_data(filename: str) -> list[dict] :
    frame_data = []
    current_frame = None

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            # Match a timestamp line, e.g. "=== Timestamp: 15.0s ==="
            #re here extracts the timestamp with any digit and it looks for the pattern in a line
            ts_match = re.match(r"=== Timestamp: ([\d.]+)s ===", line)

            #we have a time stamp match
            if ts_match:
                # Save the previous frame data before starting a new one
                if current_frame is not None:
                    frame_data.append(current_frame)
                
                #stores the current timestamp and resets the keypoints for the new frame
                current_frame = {
                    "time": ts_match.group(1) + "s",
                    "keypoints": {}
                }
                continue

            # Match a vector line, e.g. "nose -> left_eye: (-15.4, 24.8)"
            #vec_match here extracts a word -> word: (number, number) pattern and looks for it in a line
            vec_match = re.match(r"(\w+) -> (\w+): \(([-\d.]+), ([-\d.]+)\)", line)

            #if we find a match which we will as all keypoints will follow this pattern guarenteed
            if vec_match and current_frame is not None:
                #word(nose) -> word(keypoint name): (x relative to nose, y relative to nose) we extract these values and save them as variables
                nose, keypoint_name_extracted, x, y = vec_match.groups()
                x, y = float(x), float(y) #convertx and y to floats
                keypoint_name = f"{keypoint_name_extracted}"

                magnitude  = math.sqrt(x * x + y * y) #find the magnitude of current keypoint of timeframe sqrt(x^2 + y^2)
                angle_rad = math.atan2(y, x) #find the angle of current keypoint of timeframe atan2(y, x)

                #stores the keypoints for the current frame in a dictionary with the keypoint name as the key and 
                #a dictionary of magnitude and angle as the value
                current_frame["keypoints"][keypoint_name] = { 
                    "magnitude": round(magnitude,6),
                    "angle_rad": round(angle_rad,6),
                    "angle_deg":round(math.degrees(angle_rad), 6)
                }

        # Don't forget the last frame
        if current_frame is not None:
            frame_data.append(current_frame)

    return frame_data

#calculates the smallest angle difference between two angles in radians, this handles the wraparound at -pi and +pi
def angle_difference(a, b) -> float:
    # smallest angle between two radian values, handles the -pi/+pi wraparound
    diff = abs(a - b) % (2 * math.pi)
    if diff > math.pi:
        diff = 2 * math.pi - diff
    return diff

#this function takes the angle difference in degrees and converts it to a score out of 100, 
# with 10 degrees or less being a perfect score and losing 1% per degree past 15
def score_from_difference_degree(diff_deg) -> float:
    if diff_deg <= 15:
        return 100.0
    score = 100.0 - diff_deg + 15  # lose 1% per degree past 15
    if score < 0:
        score = 0.0
    return score

#main comparsion function
def compare_sequence_to_frames(sequence_order: list[str], poses: dict, frame_data: list[dict]) -> list[dict]:
    results = []

    # pair each pose name with a frame, position by position (sequentially goes through the list of poses and frames together)
    for pose_name, frame in zip(sequence_order, frame_data):
        target = poses[pose_name]       # the "correct" pose from the song
        actual = frame["keypoints"]     # what was detected at that timestamp

        worst_degree_difference = 0.0 #difference of angle for the worse joint in degrees
        worst_keypoint = None # the name of the worse joint of frame to target

        # only compare all keypoints that exist in target pose and actual frame
        # keypoint_name is the key that is just in target by now we know what pose we are looking for as it is
        # saved in pose_name from our sequence order
        for keypoint_name in target:
            if keypoint_name in actual:
                expected_value = target[keypoint_name]["angle_rad"]
                detected_value = actual[keypoint_name]["angle_rad"]
                diff = angle_difference(expected_value, detected_value)

                # keep the largest difference = the worst joint
                if diff > worst_degree_difference:
                    worst_degree_difference = diff
                    worst_keypoint = keypoint_name

            # else:
            #     # a keypoint in the target pose was not detected at all in the actual frame, this is the worst case for that joint (commented due to camera errors)
            #     worst_degree_difference = 1.75 #gives a score of 0 if no frames are detected
            #     worst_keypoint = None

        score = score_from_difference_degree(round(math.degrees(worst_degree_difference), 6))

        #saves the data of the pose name and the time frame
        #saves the worst keypoint and the difference in radians and degrees
        results.append({
            "pose_name": pose_name,
            "time": frame["time"],
            "worst_keypoint": worst_keypoint,
            "worst_degree_difference_rad": round(worst_degree_difference, 6),
            "worst_degree_difference_degree": round(math.degrees(worst_degree_difference), 6),
            "score": round(score, 0)
        })

    return results

#takes total score of all frames and averages it to get a total accuracy for the whole song performance
def total_accuracy (results: list[dict]) -> float:
    if not results:
        return 0.0
    total_score = sum(r["score"] for r in results)
    return round(total_score / len(results), 0)




def main():



    #####################################

    # #tests if the pose sequence loaded correctly or not by outputting contents to console

    # sequence_order, poses = parse_pose_sequence_data("song_example.json")

    # for pose_name in sequence_order:
    #     print(f"Pose: {pose_name}")
    #     keypoints = poses[pose_name]
    #     for name, info in keypoints.items():
    #         print(f"  {name}: dx={info['dx']}, dy={info['dy']}, magnitude={info['magnitude']}, angle_rad={info['angle_rad']}, angle_deg={info['angle_deg']}")
    #     print()

    #####################################



    #####################################

    # #tests if the frame data loaded correctly or not by outputting contents to console

    # #frame data is a list of dicts with each dict containing a timestamp and the keypoints for that timestamp
    # #the list is by time stamp as frame
    # #each frams holds the values for the keypoints data for that time stamp
    # frame_data: list[dict] = parse_frame_data("pose_locations.txt")
    # for frame in frame_data:
    #     print(frame["time"])
    #     for keypoint_name, value in frame["keypoints"].items():
    #         print(f"  {keypoint_name}: mag={value['magnitude']}, angle_rad={value['angle_rad']}, angle_deg={value['angle_deg']}")

    #####################################

    sequence_order, poses = parse_pose_sequence_data("song_example.json")
    frame_data = parse_frame_data("pose_locations.txt")

    results = compare_sequence_to_frames(sequence_order, poses, frame_data)
    # for r in results:
    #     print(f"{r['pose_name']} @ {r['time']}: "
    #         f"worst joint = {r['worst_keypoint']} "
    #         f"({r['worst_degree_difference_degree']}° off)"
    #         f"score = {r['score']}%")
        
    accuracy = total_accuracy(results)
    print(f"Total Accuracy: {accuracy}%")
            

if __name__ == "__main__":
    main()