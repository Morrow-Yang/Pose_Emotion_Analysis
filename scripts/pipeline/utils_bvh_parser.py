import numpy as np
import re

class BVHParser:
    def __init__(self, file_path):
        self.file_path = file_path
        self.joints = []
        self.frames = []
        self.frame_time = 0
        self.total_channels = 0
        self._parse()

    def _parse(self):
        with open(self.file_path, 'r') as f:
            content = f.read()

        # Split into hierarchy and motion
        parts = re.split(r'MOTION', content, flags=re.IGNORECASE)
        if len(parts) < 2:
            raise ValueError("No MOTION section found")
        
        hierarchy_text = parts[0]
        motion_text = parts[1]

        # Parse Hierarchy
        lines = hierarchy_text.splitlines()
        stack = []
        for line in lines:
            line = line.strip()
            if not line or line == '{':
                continue
            
            if line.startswith('ROOT') or line.startswith('JOINT'):
                name = line.split()[1]
                joint = {
                    'name': name,
                    'parent': stack[-1]['name'] if stack else None,
                    'offset': np.zeros(3),
                    'channels': [],
                    'children': []
                }
                self.joints.append(joint)
                if stack:
                    stack[-1]['children'].append(name)
                stack.append(joint)
            elif line.startswith('End Site'):
                joint = {
                    'name': stack[-1]['name'] + "_End",
                    'parent': stack[-1]['name'],
                    'offset': np.zeros(3),
                    'channels': [],
                    'children': []
                }
                self.joints.append(joint)
                stack[-1]['children'].append(joint['name'])
                stack.append(joint)
            elif line.startswith('OFFSET'):
                parts_p = line.split()
                stack[-1]['offset'] = np.array([float(parts_p[1]), float(parts_p[2]), float(parts_p[3])])
            elif line.startswith('CHANNELS'):
                parts_p = line.split()
                stack[-1]['channels'] = parts_p[2:]
                self.total_channels += int(parts_p[1])
            elif line == '}':
                stack.pop()

        # Parse Motion
        motion_lines = motion_text.strip().splitlines()
        num_frames = 0
        for i, m_line in enumerate(motion_lines):
            if m_line.startswith('Frames:'):
                num_frames = int(m_line.split()[1])
            elif m_line.startswith('Frame Time:'):
                self.frame_time = float(m_line.split()[2])
                # The rest is data
                data_text = " ".join(motion_lines[i+1:])
                all_values = [float(x) for x in data_text.split()]
                
                # Check if total values match num_frames * total_channels
                if len(all_values) < num_frames * self.total_channels:
                    num_frames = len(all_values) // self.total_channels
                
                for f in range(num_frames):
                    start = f * self.total_channels
                    end = start + self.total_channels
                    self.frames.append(all_values[start:end])
                break

    def get_joint_world_coords(self, frame_idx):
        """Returns a dictionary {joint_name: [x, y, z]} for a given frame."""
        if frame_idx >= len(self.frames):
            return {}
            
        data = self.frames[frame_idx]
        world_coords = {}
        rotations = {} 
        
        data_idx = 0
        for joint in self.joints:
            local_pos = joint['offset'].copy()
            local_rot_matrix = np.eye(3)
            
            for channel in joint['channels']:
                val = data[data_idx]
                data_idx += 1
                
                c_low = channel.lower()
                if c_low == 'xposition': local_pos[0] = val
                elif c_low == 'yposition': local_pos[1] = val
                elif c_low == 'zposition': local_pos[2] = val
                else:
                    angle = np.radians(val)
                    c, s = np.cos(angle), np.sin(angle)
                    if c_low == 'xrotation': m = np.array([[1,0,0],[0,c,-s],[0,s,c]])
                    elif c_low == 'yrotation': m = np.array([[c,0,s],[0,1,0],[-s,0,c]])
                    elif c_low == 'zrotation': m = np.array([[c,-s,0],[s,c,0],[0,0,1]])
                    else: continue
                    local_rot_matrix = local_rot_matrix @ m

            parent_name = joint['parent']
            if parent_name is None:
                world_coords[joint['name']] = local_pos
                rotations[joint['name']] = local_rot_matrix
            else:
                parent_world_pos = world_coords[parent_name]
                parent_world_rot = rotations[parent_name]
                world_coords[joint['name']] = parent_world_pos + parent_world_rot @ local_pos
                rotations[joint['name']] = parent_world_rot @ local_rot_matrix
                
        return world_coords

if __name__ == "__main__":
    # Quick Test
    import sys
    test_file = r"data/raw/kinematic-dataset-of-actors-expressing-emotions-2.1.0/BVH/F01/F01A0V1.bvh"
    parser = BVHParser(test_file)
    coords = parser.get_joint_world_coords(0)
    print(f"Frame 0, Head: {coords.get('Head')}")
    print(f"Total frames: {len(parser.frames)}")

