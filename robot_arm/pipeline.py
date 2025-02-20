from robot_arm.command import Command


class Pipeline:
    """Implementation of pipeline pattern."""
    def __init__(self):
        self.commands = []

    def add_command(self, command: Command):
        """Add command to the tail of the pipeline."""
        self.commands.append(command)

    def add_command_reverse(self, command: Command):
        """Add command to the head of the pipeline."""
        self.commands.insert(0, command)

    def run(self):
        """Perform each command.execute() in order."""
        for command in self.commands:
            command.execute()

    def undo(self):
        """Perform each command.undo() in reversed order."""
        self.commands.pop()
        for command in reversed(self.commands):  # 逆序执行 undo
            command.undo()
