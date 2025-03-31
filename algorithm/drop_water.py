from math import *

L = 150
alpha = radians(10)
beta = radians(30)

def convert_teacup_to_teapot(xd, yd, zd, theta=radians(-45)):
    P = L * cos(theta)
    Q = L * (1 - cos(theta)) * cos(alpha) * cos(beta)
    R = L * sin(theta)
    u = P * cos(beta) + Q * cos(alpha) + R * sin(alpha) * cos(beta)
    v = P * sin(beta) - R * sin(alpha) * cos(beta)
    w = sqrt(xd ** 2 + yd ** 2 - v ** 2)
    r = w - u
    x = r * (w * xd + v * yd) / (w ** 2 + v ** 2)
    y = r * (w * yd - v * xd) / (w ** 2 + v ** 2)
    z = zd + Q * sin(alpha) - R * cos(alpha) * sin(beta)
    return [x, y, z]


if __name__ == '__main__':
    x_cup, y_cup, z_cup = 100, 100, 44
    print(convert_teacup_to_teapot(x_cup, y_cup, z_cup))
