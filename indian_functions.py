import math as m

import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import griddata


def getChannellInfoForSample(channelNames, channelValues, onlyValues=False):
    i = 0
    channelValuesforCurrentSample = []
    for ch in channelNames:
        chValue = channelValues[i]
        if onlyValues:
            channelValuesforCurrentSample.append(chValue)
        else:
            channelValuesforCurrentSample.append((ch, chValue))
        i += 1

    return channelValuesforCurrentSample


def convert3DTo2D(pos_3d):
    pos_2d = []
    for e in pos_3d:
        pos_2d.append(azim_proj(e))

    return (pos_2d)


def cart2sph(x, y, z):
    """
    Transform Cartesian coordinates to spherical
    :param x: X coordinate
    :param y: Y coordinate
    :param z: Z coordinate
    :return: radius, elevation, azimuth
    """
    x2_y2 = x ** 2 + y ** 2
    r = m.sqrt(x2_y2 + z ** 2)  # r     tant^(-1)(y/x)
    elev = m.atan2(z, m.sqrt(x2_y2))  # Elevation
    az = m.atan2(y, x)  # Azimuth

    return r, elev, az


def azim_proj(pos):
    """
    Computes the Azimuthal Equidistant Projection of input point in 3D Cartesian Coordinates.
    Imagine a plane being placed against (tangent to) a globe. If
    a light source inside the globe projects the graticule onto
    the plane the result would be a planar, or azimuthal, map
    projection.

    :param pos: position in 3D Cartesian coordinates    [x, y, z]
    :return: projected coordinates using Azimuthal Equidistant Projection
    """
    [r, elev, az] = cart2sph(pos[0], pos[1], pos[2])

    return pol2cart(az, m.pi / 2 - elev)


def pol2cart(theta, rho):
    """
    Transform polar coordinates to Cartesian
    :param theta: angle value
    :param rho: radius value
    :return: X, Y
    """
    # print ('----------')
    # print (rho * m.cos(theta))
    # print (rho * m.sin(theta))
    return rho * m.cos(theta), rho * m.sin(theta)


def get3DCoordinates(MontageChannelLocation, EEGChannels):
    MontageChannelLocation = MontageChannelLocation[-EEGChannels:]
    location = []
    for i in range(0, 32):
        v = MontageChannelLocation[i].values()
        values = list(v)
        a = (values[1]) * 1000
        location.append(a)
    MontageLocation = np.array(location)
    # MontageLocation=trunc(MontageLocation,decs=3)
    MontageLocation = np.round(MontageLocation, 1)
    MontageLocation = MontageLocation.tolist()
    return MontageLocation


def createTopographicMapFromChannelValues(channelValues, rawDatasetReReferenced, interpolationMethod="cubic",
                                          verbose=False):
    channelNames = ['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5', 'P7',
                    'P3', 'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8', 'CP6', 'CP2',
                    'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'AF4', 'Fp2', 'Fz', 'Cz']
    listOfChannelValues = getChannellInfoForSample(channelNames, channelValues, onlyValues=True)

    rawDatasetForMontageLocation = rawDatasetReReferenced.copy()
    MontageChannelLocation = rawDatasetForMontageLocation.info['dig']
    lengthOfTopographicMap = 40  # 40x40
    emptyTopographicMap = np.array(np.zeros([lengthOfTopographicMap, lengthOfTopographicMap]))

    if verbose:
        plt.imshow(emptyTopographicMap)
        plt.show()

    NumberOfEEGChannel = 32
    pos2D = np.array(convert3DTo2D(get3DCoordinates(MontageChannelLocation, NumberOfEEGChannel)))

    grid_x, grid_y = np.mgrid[
                     min(pos2D[:, 0]):max(pos2D[:, 0]):lengthOfTopographicMap * 1j,
                     min(pos2D[:, 1]):max(pos2D[:, 1]):lengthOfTopographicMap * 1j
                     ]

    # Generate edgeless images

    min_x, min_y = np.min(pos2D, axis=0)
    max_x, max_y = np.max(pos2D, axis=0)
    locations = np.append(pos2D, np.array([[min_x, min_y], [min_x, max_y], [max_x, min_y], [max_x, max_y]]), axis=0)

    interpolatedTopographicMap = griddata(pos2D, channelValues, (grid_x, grid_y), method=interpolationMethod,
                                          fill_value=0)

    CordinateYellowRegion = np.argwhere(interpolatedTopographicMap == 0.)

    if verbose:
        i = 0
        for chVal in channelValues:
            for x in range(32):
                for y in range(32):
                    print("Trying to find value {} in pixel ({},{}-{})".format(chVal, x, y,
                                                                               interpolatedTopographicMap[x][y]))
                    if (chVal == interpolatedTopographicMap[x][y]):
                        print("Value found at pixel ({}{}) for channel: {}".format(x, y, channelNames[i]))
            i = i + 1

    if (verbose):
        plt.imshow(interpolatedTopographicMap)
        plt.show()

    return interpolatedTopographicMap, CordinateYellowRegion
