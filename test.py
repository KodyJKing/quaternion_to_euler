from random import Random
import unittest
import numpy as np

from main import euler_matrix, euler_quat, matrix_to_euler, quat_to_euler

DEGREES = np.pi/180
RADIANS = 180/np.pi

r = Random()
r.seed("This is the quaternion to euler angles RNG-seed.")
np.set_printoptions(precision=3)

test_iterations = 1000
degree_tolerance = 1e-6

def random_euler_angles():
    return np.array([
        r.uniform(-180, 180)*DEGREES, 
        r.uniform(-90,  90 )*DEGREES,
        r.uniform(-180, 180)*DEGREES
    ])

def compare_angles(test: unittest.TestCase, angles_in, angles_out):
    angle_errors = np.abs((angles_in-angles_out)*RADIANS)
    # print("   In: ", angles_in*RADIANS)
    # print("   Out:", angles_out*RADIANS)
    # print("   Err:", angle_errors)
    for e in angle_errors:
        test.assertTrue(e < degree_tolerance)

class TestToEulerFunctions(unittest.TestCase):

    def test_matrix_to_euler(self):
        for i in range(test_iterations):
            angles_in = random_euler_angles()
            m = euler_matrix(*angles_in)
            angles_out = matrix_to_euler(m)
            compare_angles(self, angles_in, angles_out)

    def test_quat_to_euler(self):
        for i in range(test_iterations):
            angles_in = random_euler_angles()
            q = euler_quat(*angles_in)
            angles_out = quat_to_euler(q)
            compare_angles(self, angles_in, angles_out)

if __name__ == "__main__":
    unittest.main()