import numpy as np

def transform(image,
              translation=(0, 0, 0),
              rotation=(0, 0, 0),
              scaling=(1, 1, 1),
              shearing=(0, 0, 0)):
    # get the values on each axis
    t_x, t_y, t_z = translation
    r_x, r_y, r_z = rotation
    sc_x, sc_y, sc_z = scaling
    sh_x, sh_y, sh_z = shearing

    # convert degree angles to rad
    theta_rx = np.deg2rad(r_x)
    theta_ry = np.deg2rad(r_y)
    theta_rz = np.deg2rad(r_z)
    theta_shx = np.deg2rad(sh_x)
    theta_shy = np.deg2rad(sh_y)
    theta_shz = np.deg2rad(sh_z)

    # get the height and the width of the image
    w = image[0]
    h = image[1]

    # calculate cos and sin of angles
    sin_rx, cos_rx = np.sin(theta_rx), np.cos(theta_rx)
    sin_ry, cos_ry = np.sin(theta_ry), np.cos(theta_ry)
    sin_rz, cos_rz = np.sin(theta_rz), np.cos(theta_rz)

    # get the rotation matrix on x axis
    R_Mx = np.array([[1, 0, 0, 0],
                     [0, cos_rx, -sin_rx, 0],
                     [0, sin_rx, cos_rx, 0],
                     [0, 0, 0, 1]])
    # get the rotation matrix on y axis
    R_My = np.array([[cos_ry, 0, -sin_ry, 0],
                     [0, 1, 0, 0],
                     [sin_ry, 0, cos_ry, 0],
                     [0, 0, 0, 1]])
    # get the rotation matrix on z axis
    R_Mz = np.array([[cos_rz, -sin_rz, 0, 0],
                     [sin_rz,  cos_rz, 0, 0],
                     [0,  0, 1, 0],
                     [0, 0, 0, 1]])
    # compute the full rotation matrix
    R_M = np.dot(np.dot(R_Mx, R_My), R_Mz)
    # get the scaling matrix
    Sc_M = np.array([[sc_x, 0, 0, 0],
                     [0, sc_y, 0, 0],
                     [0, 0, sc_z, 0],
                     [0, 0,  0, 1]])

    # get the tan of angles
    tan_shx = np.tan(theta_shx)
    tan_shy = np.tan(theta_shy)
    tan_shz = np.tan(theta_shz)
    # get the shearing matrix on x axis
    Sh_Mx = np.array([[1, 0, 0, 0],
                      [tan_shy, 1, 0, 0],
                      [tan_shz, 0, 1, 0],
                      [0, 0, 0, 1]])
    # get the shearing matrix on y axis
    Sh_My = np.array([[1, tan_shx, 0, 0],
                      [0, 1, 0, 0],
                      [0, tan_shz, 1, 0],
                      [0, 0, 0, 1]])
    # get the shearing matrix on z axis
    Sh_Mz = np.array([[1, 0, tan_shx, 0],
                      [0, 1, tan_shy, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])
    # compute the full shearing matrix
    Sh_M = np.dot(np.dot(Sh_Mx, Sh_My), Sh_Mz)

    Identity = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])

    # compute the full transform matrix
    M = Identity
    M = np.dot(Sc_M, M)
    M = np.dot(Sh_M, M)
    # apply the transformation
    temp_matrix = np.array([[w],
                            [h],
                            [1],
                            [1]])
    new_coordinates = np.dot(M, temp_matrix)
    new_x = int(new_coordinates[0][0])
    new_y = int(new_coordinates[1][0])
    # return new coordinates
    return (new_x, new_y)