from random import Random
import unittest
import numpy as np
from main import euler_matrix, euler_quat, matmul_n, matrix_to_euler, pitch_matrix, quat_to_euler, roll_matrix, yaw_matrix

np.set_printoptions(precision=3, suppress=True)

DEGREES = np.pi/180
RADIANS = 180/np.pi

r = Random()
r.seed("This is the quaternion to euler angles RNG-seed.")

test_iterations = 1000
degree_tolerance = 1e-6
linear_tolerance = 1e-6

def random_euler_angles():
    return np.array([
        r.uniform(-180, 180)*DEGREES, 
        r.uniform(-90,  90 )*DEGREES,
        r.uniform(-180, 180)*DEGREES
    ])

def compare_angles(test: unittest.TestCase, angles_in, angles_out):
    angle_errors = np.abs((angles_in-angles_out)*RADIANS)
    for e in angle_errors:
        if e >= degree_tolerance:
            print("\n")
            print("In: ", angles_in*RADIANS)
            print("Out:", angles_out*RADIANS)
            print("Err:", angle_errors)
            print()
            test.fail()
        
def compare_angles_by_mat(test: unittest.TestCase, angles_in, angles_out):
    mat_in = euler_matrix(*angles_in)
    mat_out = euler_matrix(*angles_out)
    mat_err = mat_out - mat_in
    for row in mat_err:
        for v in row:
            if np.abs(v) >= linear_tolerance:
                print("\n")
                print("In Angles: ", angles_in*RADIANS)
                print("Out Angles:", angles_out*RADIANS)
                print("In Matrix:\n", mat_in)
                print("Out Matrix:\n", mat_out)
                print("Err Matrix:\n", mat_err)
                print()
                test.fail()

class TestToEulerFunctions(unittest.TestCase):

    def test_matrix_to_euler(self):
        for i in range(test_iterations):
            angles_in = random_euler_angles()
            m = euler_matrix(*angles_in)
            angles_out = matrix_to_euler(m)
            # compare_angles(self, angles_in, angles_out)
            compare_angles_by_mat(self, angles_in, angles_out)

    def test_quat_to_euler(self):
        for i in range(test_iterations):
            angles_in = random_euler_angles()
            q = euler_quat(*angles_in)
            angles_out = quat_to_euler(q)
            # compare_angles(self, angles_in, angles_out)
            compare_angles_by_mat(self, angles_in, angles_out)

    def test_matrix_to_euler_gimble_lock(self):
        rotations = [
            np.array([90*DEGREES, 90*DEGREES, 0*DEGREES]),
            np.array([90*DEGREES, -90*DEGREES, 0*DEGREES]),
            np.array([45*DEGREES, 90*DEGREES, 0*DEGREES]),
            np.array([45*DEGREES, -90*DEGREES, 0*DEGREES]),
            np.array([45*DEGREES, 90*DEGREES, 45*DEGREES]),
            np.array([45*DEGREES, -90*DEGREES, 45*DEGREES]),
        ]
        for angles_in in rotations:
            yaw, pitch, roll = angles_in
            m = matmul_n(
                yaw_matrix(yaw),
                 # Round pitch matrix so we don't leak any yaw information into the first column.
                np.round(pitch_matrix(pitch)),
                roll_matrix(roll),
            )
            # print(m)
            angles_out = matrix_to_euler(m)
            compare_angles_by_mat(self, angles_in, angles_out)

if __name__ == "__main__":
    unittest.main()