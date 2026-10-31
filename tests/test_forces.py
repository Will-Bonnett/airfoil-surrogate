import pytest
import numpy as np
from forces import pressure_force_coefficients, pressure_force

def test_constant_pressure_zeros():
    """
    Build a square with constant pressure,    
    The total force should be zero

    Square layout:
    1   3

    0   2
    """
    points = np.array((
        [0,0],
        [0,1],
        [1,0],
        [1,1]))
    # Segments must be in a closed loop
    segments = np.array((
        [0,1],
        [1,3],
        [3,2],
        [2,0]))
    pressures = np.array((
        10.0,
        10.0,
        10.0,
        10.0))

    Fx, Fy = pressure_force(points, segments, pressures)
    
    assert Fx == 0
    assert Fy == 0
    

def test_cardinal_pressure():
    """
    Build a square with pressure on one side
    in this case both points 1 and 3 have pressure 5;
    as a result stick (1,3) has stick pressure 5 pushing down
    causing a Fy of -5
    
    thus normal = force
    
    Square layout:
    1   3

    0   2
    """
    points = np.array((
        [0,0],
        [0,1],
        [1,0],
        [1,1]))
    segments = np.array((
        [0,1],
        [1,3],
        [3,2],
        [2,0]))
    pressures = np.array((
        0.0,
        5.0,
        0.0,
        5.0))

    Fx, Fy = pressure_force(points, segments, pressures)
    print(Fx,Fy)

    # allclose to allow for potential floating point errors
    assert np.allclose([Fx,Fy],[0,-5])