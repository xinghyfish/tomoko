import logging
import time

from robot_arm.command import Command

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为 DEBUG
    format='%(asctime)s - %(levelname)s - %(message)s',
)

class Pipeline:
    """Implementation of pipeline pattern."""
    def __init__(self, label: str):
        self.label = label
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
            flag = command.execute()
            time.sleep(1)
            if flag:
                logging.info(f"{command} execute done.")
            else:
                logging.error(f"{command} execute failed.")
                return False
        logging.info(f"Pipeline [{self.label}] execution is done.")
        return True

    def undo(self):
        """Perform each command.undo() in reversed order."""
        for command in reversed(self.commands):  # 逆序执行 undo
            flag = command.undo()
            time.sleep(1)
            if flag:
                logging.info(f"{command} undo done.")
            else:
                logging.error(f"{command} undo failed.")
                return False
        logging.info(f"Pipeline [{self.label}] undo is done.")
        return True

    def __str__(self):
        msg = f"[Pipeline] {self.label}\n"
        for command in self.commands:
            msg += f"\t{command}\n"
