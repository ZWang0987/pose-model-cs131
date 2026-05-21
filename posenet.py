#!/usr/bin/env python3
#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#


#############################Working extractor#############################


#!/usr/bin/env python3
import sys
import argparse
import time
from jetson_inference import poseNet
from jetson_utils import videoSource, videoOutput, Log

parser = argparse.ArgumentParser(description="Run pose estimation DNN on a video/image stream.", 
    formatter_class=argparse.RawTextHelpFormatter, 
    epilog=poseNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())
parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="resnet18-body", help="pre-trained model to load")
parser.add_argument("--overlay", type=str, default="links,keypoints", help="pose overlay flags")
parser.add_argument("--threshold", type=float, default=0.15, help="minimum detection threshold to use")
parser.add_argument("--output-file", type=str, default="pose_locations.txt", help="output text file for pose data")

try:
    args = parser.parse_known_args()[0]
except:
    print("")
    parser.print_help()
    sys.exit(0)

net = poseNet(args.network, sys.argv, args.threshold)

#for video input and output
input = videoSource(args.input, argv=sys.argv)
output = videoOutput(args.output, argv=sys.argv)

#restart the clock and work on 5 second intervals
last_save_time = time.time()
SAVE_INTERVAL = 5.0
start_time = time.time()

# Clear the file at the start of each run
with open(args.output_file, 'w') as f:
    f.write("")

while True:
    img = input.Capture()
    if img is None:
        continue

    #here poses is what pose net is using for processing and not a actual pose vector
    poses = net.Process(img, overlay=args.overlay)

    current_time = time.time()
    if current_time - last_save_time >= SAVE_INTERVAL:
        elapsed = current_time - start_time 
        with open(args.output_file, 'a') as f:
            f.write(f"\n=== Timestamp: {elapsed:.1f}s ===\n")

            #no one on the screen at all
            if not poses:
                f.write("  No poses detected\n")

            #someone detected
            for pose in poses:

                #orginal cordinates which I think will not be needed as all we need is the directional vector
                # f.write(f"  Pose {pose.ID}:\n")
                # for kp in pose.Keypoints:
                #     name = net.GetKeypointName(kp.ID)  # FIX 1: replaces kp.Description
                #     f.write(f"    {name}: x={kp.x:.1f}, y={kp.y:.1f}\n")


                # Find neck point
                neck_kp = None
                for kp in pose.Keypoints:
                    if net.GetKeypointName(kp.ID) == "neck":
                        neck_kp = kp
                        break

                # Vectors from neck to all other keypoints
                if neck_kp is not None:
                    f.write("    Direction Vectors (relative to neck):\n")
                    for kp in pose.Keypoints:
                        name = net.GetKeypointName(kp.ID)
                        if name == "neck":
                            continue
                        vec_x = neck_kp.x - kp.x
                        vec_y = neck_kp.y - kp.y
                        f.write(f"      neck -> {name}: ({vec_x:.1f}, {vec_y:.1f})\n")
                else:
                    f.write("    Neck not detected, skipping direction vectors\n")
                    
        last_save_time = current_time

    print("detected {:d} objects in image".format(len(poses)))
    for pose in poses:
        print(pose)
        print(pose.Keypoints)
        print('Links', pose.Links)

    output.Render(img)
    output.SetStatus("{:s} | Network {:.0f} FPS storage test-1".format(args.network, net.GetNetworkFPS()))
    net.PrintProfilerTimes()

    if not input.IsStreaming() or not output.IsStreaming():
        break






##########################################################################
#                           ORIGINAL POSENET MODEL
# import sys
# import argparse

# from jetson_inference import poseNet
# from jetson_utils import videoSource, videoOutput, Log

# # parse the command line
# parser = argparse.ArgumentParser(description="Run pose estimation DNN on a video/image stream.", 
#                                  formatter_class=argparse.RawTextHelpFormatter, 
#                                  epilog=poseNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

# parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
# parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
# parser.add_argument("--network", type=str, default="resnet18-body", help="pre-trained model to load (see below for options)")
# parser.add_argument("--overlay", type=str, default="links,keypoints", help="pose overlay flags (e.g. --overlay=links,keypoints)\nvalid combinations are:  'links', 'keypoints', 'boxes', 'none'")
# parser.add_argument("--threshold", type=float, default=0.15, help="minimum detection threshold to use") 

# try:
# 	args = parser.parse_known_args()[0]
# except:
# 	print("")
# 	parser.print_help()
# 	sys.exit(0)

# # load the pose estimation model
# net = poseNet(args.network, sys.argv, args.threshold)

# # create video sources & outputs
# input = videoSource(args.input, argv=sys.argv)
# output = videoOutput(args.output, argv=sys.argv)

# # process frames until EOS or the user exits
# while True:
#     # capture the next image
#     img = input.Capture()

#     if img is None: # timeout
#         continue  

#     # perform pose estimation (with overlay)
#     poses = net.Process(img, overlay=args.overlay)

#     # print the pose results
#     print("detected {:d} objects in image".format(len(poses)))

#     for pose in poses:
#         print(pose)
#         print(pose.Keypoints)
#         print('Links', pose.Links)

#     # render the image
#     output.Render(img)

#     # update the title bar
#     output.SetStatus("{:s} | Network {:.0f} FPS goated".format(args.network, net.GetNetworkFPS()))

#     # print out performance info
#     net.PrintProfilerTimes()

#     # exit on input/output EOS
#     if not input.IsStreaming() or not output.IsStreaming():
#         break
