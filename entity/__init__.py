class GraspInfo:
    def __init__(self, target_index=None, end_pose=None):
        self.position = target_index
        self.end_pose = end_pose

    def __repr__(self):
        return f"#position = {self.position}, #end_pose = {self.end_pose}"