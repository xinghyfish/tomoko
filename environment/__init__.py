front_table_height = 97.9
side_table_height = 225.0
under_board = -1.0


class GraspInfo:
    def __init__(self, target_index=None, end_pose=None):
        self.position = target_index
        self.end_pose = end_pose

    def __repr__(self):
        return f"#position = {self.position}, #end_pose = {self.end_pose}"
