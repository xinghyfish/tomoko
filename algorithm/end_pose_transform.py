from numpy import arctan

from robot_arm.arm_controller import hand_length, arm_radius
import math

def end_pose_transform(x, y, z, distance, theta, delta_h):
    """
    Transform detected point to end pose to grasp it.
    :param x: x coordinate of target position
    :param y: y coordinate of target position
    :param z: z coordinate of target position
    :param distance: distance of surface to grasp point
    :param theta: vertical angle of grasp pose
    :param delta_h: distance of vertical lift of end
    :return: end pose to grasp target object
    """
    # You may find it like magic.
    # Yes, indeed. math is just like magic. \【T】/
    Rx, Ry, Rz = 0, theta, math.degrees(float(arctan(y / x)))
    theta = math.radians(theta)
    l = hand_length
    mult_factor = (distance - l) * math.sin(theta) / math.sqrt(math.pow(x, 2) + math.pow(y, 2))
    x_e = (1 + mult_factor) * x
    y_e = (1 + mult_factor) * y
    z_e = z + (distance - l) * math.cos(theta) + (delta_h / math.sin(theta))

    return [x_e, y_e, z_e, Rx, Ry, Rz]


def bottom_transform(target_index, distance, theta):
    x, y, z = target_index
    return end_pose_transform(x, y, z, distance, theta, arm_radius * 2)


def middle_transform(target_index, distance, theta):
    x, y, z = target_index
    return end_pose_transform(x, y, z, distance, theta, arm_radius)


def top_transform(target_index, distance, theta):
    x, y, z = target_index
    return end_pose_transform(x, y, z, distance, theta, 0)