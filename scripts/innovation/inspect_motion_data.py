import numpy as np
from scripts.innovation.utils_bvh_parser import BVHParser

test_file = r"data/raw/kinematic-dataset-of-actors-expressing-emotions-2.1.0/BVH/F01/F01A0V1.bvh"
parser = BVHParser(test_file)

print(f"Joint: {parser.joints[0]['name']} | Channels: {parser.joints[0]['channels']}")
print(f"Motion Data for Frame 0 (first 12 values): {parser.frames[0][:12]}")
print(f"Joint: {parser.joints[1]['name']} | Channels: {parser.joints[1]['channels']}")
