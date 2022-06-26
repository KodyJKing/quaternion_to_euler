from functools import reduce
import numpy as np

DEFAULT_GIMBLE_LOCK_THESHOLD = 1e-7

#region Helpers

def cos_sin(theta):
    return np.cos(theta), np.sin(theta)

def normalized(array: np.array):
    return array / np.linalg.norm(array)

def matmul_n(*mats):
    return reduce(lambda a, b: np.matmul(a, b), mats)

#endregion

#region Euler to Matrix

def roll_matrix(theta):
    c, s = cos_sin(theta)
    return np.array([
        [1, 0,  0],
        [0, c, -s],
        [0, s,  c]
    ])
def pitch_matrix(theta):
    c, s = cos_sin(theta)
    return np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c]
    ])
def yaw_matrix(theta):
    c, s = cos_sin(theta)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])
def euler_matrix(yaw, pitch, roll):
    return np.matmul(
        yaw_matrix(yaw),
        np.matmul(
            pitch_matrix(pitch),
            roll_matrix(roll)
        )
    )

#endregion

#region Matrix to Euler 

def matrix_to_euler(m, gimble_lock_threshold=DEFAULT_GIMBLE_LOCK_THESHOLD):
    mxx = m[0][0]
    mxy = m[1][0]
    mxz = m[2][0]

    if mxx**2 + mxy**2 < gimble_lock_threshold:
        mzx = m[0][2]
        mzy = m[1][2]
        return matrix_to_euler_gimble_locked(mxz, mzx, mzy)
    
    myz = m[2][1]
    mzz = m[2][2]

    return matrix_to_euler_standard(mxx, mxy, mxz, myz, mzz)

def matrix_to_euler_gimble_locked(mxz, mzx, mzy):
    sgn_pitch = np.sign(-mxz)
    cos_yaw = mzx * sgn_pitch
    sin_yaw = mzy * sgn_pitch
    yaw = np.arctan2(sin_yaw, cos_yaw)
    return np.array([yaw, np.pi/2 * sgn_pitch, 0])

def matrix_to_euler_standard(mxx, mxy, mxz, myz, mzz):
    # Yaw is the angle column-x.xy makes the x-axis. (+y is positive)
    yaw = np.arctan2(mxy, mxx)
    # Pitch is the angle column-x makes with the xy-plane. (-z is positive)
    mx_horizontal = np.sqrt(mxx**2 + mxy**2)
    pitch = np.arctan2(-mxz, mx_horizontal)
    # Roll is the angle row-z.yz makes with the z-axis. (+y is positive)
    roll = np.arctan2(myz, mzz)
    return np.array([yaw, pitch, roll])

#endregion

#region Quaternions

def axis_angle_quat(axis: np.array, angle: float):
    c, s = cos_sin(angle/2)
    n = normalized(axis)
    return np.array([c, *(s * n)])

def quat_multiply(q1, q2):
    a1, b1, c1, d1 = q1
    a2, b2, c2, d2 = q2
    return np.array([
        a1*a2 - b1*b2 - c1*c2 - d1*d2,
        a1*b2 + b1*a2 + c1*d2 - d1*c2,
        a1*c2 - b1*d2 + c1*a2 + d1*b2,
        a1*d2 + b1*c2 - c1*b2 + d1*a2
    ])

def quat_multiply_n(*quats):
    return reduce(lambda a, b: quat_multiply(a, b), quats)

# def quat_conj(q):
#     a, b, c, d = q
#     return np.array([a, -b, -c, -d])

# def quat_rotate(q, v):
#     return quat_multiply_n(q, np.array([0, *v]), quat_conj(q))[1:]

#endregion

#region Euler to Quaternion 

def roll_quat(theta):
    return axis_angle_quat(np.array([1, 0, 0]), theta)
def pitch_quat(theta):
    return axis_angle_quat(np.array([0, 1, 0]), theta)
def yaw_quat(theta):
    return axis_angle_quat(np.array([0, 0, 1]), theta)
def euler_quat(yaw, pitch, roll):
    return quat_multiply_n(
        yaw_quat(yaw),
        pitch_quat(pitch),
        roll_quat(roll)
    )

#endregion

#region Quaternion to Euler

def quat_to_euler(q, gimble_lock_threshold=DEFAULT_GIMBLE_LOCK_THESHOLD):
    a, b, c, d = q
    # Get needed components from quaternion's rotation matrix.
    # https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Conversion_to_and_from_the_matrix_representation
    mxx = a*a + b*b - c*c - d*d
    mxy = 2*b*c + 2*a*d
    mxz = 2*b*d - 2*a*c

    if mxx**2 + mxy**2 < gimble_lock_threshold:
        mzx = 2*b*d + 2*a*c
        mzy = 2*c*d - 2*a*b
        return matrix_to_euler_gimble_locked(mxz, mzx, mzy)
    
    myz = 2*c*d + 2*a*b
    mzz = a*a -b*b -c*c + d*d
    
    return matrix_to_euler_standard(mxx, mxy, mxz, myz, mzz)

#endregion
