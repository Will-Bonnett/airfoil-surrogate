
def reynolds_number(velocity, chord=1.0, nu=1.56e-5):
    """
    Reynolds number: ratio of inertial (momentum) to viscous effects in the flow

        Re = U * c / nu

    Low Re -> viscosity dominates, smooth flow.
    High Re -> viscous effects are confined to a thin boundary layer near the surface.
                
    Flows over same shape at the same Re behave the same, so Re (with shape and angle
    of attack) determines Cl and Cd.

    Args:
        - Velocity: freestream speed U (m/s)
        - Chord: reference length c (m): 1 for all airfrans airfoils
        - nu:   kinematic viscosity of air (m^2/s). Default 1.56e-5 is the value
                AirFrans was generated with (per its docs); the library's own
                temperature formula gives ~1.55e-5

    Returns:
        - Re (dimensionless)
    """
    return velocity * chord / nu