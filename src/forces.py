import math
import numpy as np

def pressure_force_coefficients(simulation):
    """
    Returns Pressure-only drag and lift coefficents, computed by integrating
    surface pressure around the airfoil outline

    The outline is split into straight segments ("sticks"). Force on each stick 
    = average pressure x length x inward unit vector normal; these are summed,
    rotated by the angle of attack, and non-dimensionalised by 1/2 U^2 c.

    Validated agains simulation.force_coefficient() on one case:
    cl_p within 0.002%, cd_p within 9%.

    Args:
        simulation: an airfrans.Simulation

    Returns:
        (cd_p,cl_p): pressure parts of the lift and drag coefficients
        Excludes the viscous (skin friction) contribution
    """
    chord = 1.0 # length of chord (all 1 in airfrans)
    aoa = simulation.angle_of_attack # In radians
    dymprs = 0.5 * simulation.inlet_velocity**2 # Dynamic pressure value 

    # get mesh data for simulation
    af_mesh = simulation.airfoil

    # Build table of sticks
    table = af_mesh.lines.reshape(-1,3)
    table = table[:,1:]
    
    fx, fy = pressure_force(af_mesh.points[:,:2], table, af_mesh.point_data["p"])

    # rotate (Fx,Fy) from chord axes into wind axes:
    # - Drag along the freestream
    # - Lift perpendicular
    drag = fx*math.cos(aoa) + fy*math.sin(aoa)
    lift = -fx*math.sin(aoa) + fy*math.cos(aoa)

    # Calculate coefficents
    cl_p = lift / (dymprs * chord)
    cd_p = drag / (dymprs * chord)

    return (cd_p, cl_p)


def pressure_force(points, segments, p):
    """
    Returns the total pressure force

    Args:
        - points (N,2): x,y of each surface point
        - segments (M, 2): start / end point indices of each stick
        - p (N,): pressure at each point
    
    Returns:
        total_force (Fx, Fy)
    """
    # Get coordinates of points
    coords_A = points[segments[:,0]]
    coords_B = points[segments[:,1]]
    
    # Get Lengths
    d = coords_A - coords_B # difference
    lengths = np.linalg.norm(d, axis = 1)

    # get unit vector of stick normal inwards
    unit = d / lengths[:,None] # Form (dx, dy)
    normal = np.column_stack((-unit[:,1], unit[:,0]))

    # get stick pressure
    p_stick = (
        p[segments[:,0]] +
        p[segments[:,1]]
    ) / 2

    # calculate force per stick
    #   - force = normal * pressure * length
    force = normal * (p_stick * lengths)[:,None]

    return force.sum(axis=0)