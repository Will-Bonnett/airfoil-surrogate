import pandas as pd
import numpy as np
from airfrans.naca_generator import camber_line

def add_geometry_features(df):
    """
    Takes in the dataframe of airfoil data and creates the new columns of the geometry data

    Args:
        - df: DataFrame of airfoil data

    Returns
        - df: DataFrame with new columns
    """
    new_df = pd.DataFrame([camber_geometry(m,p,xx,l,q) for m,p,xx,l,q in zip(df["M"],df["P"],df["XX"],df["L"],df["Q"])], index=df.index)
    df = pd.concat([df, new_df], axis=1)

    return df

def camber_geometry(M,P,XX,L,Q):
    """
    Args:
        - M:  4-digit only. Maximum camber, in % of chord (e.g. 2 -> 2%).
        - P:  Position of maximum camber. 4-digit: in tenths of chord (4 -> 0.4c).
              5-digit: design camber position in twentieths (P/20).
        - XX: Maximum thickness, in % of chord (both families).
        - L:  5-digit only. Design lift coefficient parameter (design Cl = 0.15 x L).
        - Q:  5-digit only. Reflex flag: 0 = normal camber, 1 = reflex (tail turns up).
        
        For each family, the other family's parameters are NaN.

    Returns:
        dict with max_camber, camber_pos, thickness (all as fractions of chord,
        e.g. 0.02 not 2), and reflex (0 or 1).
    """
    # Decide if its a 4 or 5 digit airfoil
    if(pd.isna(L) and pd.isna(Q)):
        # M, P, XX (4 digit airfoil)
        max_camber = M/100
        position = P/10
        thickness = XX/100
        reflex = 0
    elif(pd.isna(M)):
        # L,P,Q,XX (5 digit airfoil)
        intervals = np.linspace(0,1,1001)
        line = camber_line([L,P,int(Q)],intervals)

        max_camber = np.max(line[0])
        position = intervals[np.argmax(line[0])]
        thickness = XX/100
        reflex = int(Q)
    else:
        raise ValueError(f"Inputted Data Does not fit digit structure: M={M} P={P} XX={XX} L={L} Q={Q}")

    return {
        "max_camber": max_camber, 
        "camber_pos": position,
        "thickness": thickness,
        "reflex": reflex}


if __name__ == "__main__":
    test_df = pd.DataFrame({
        "name": ["NACA 2412", "NACA 0012", "5-digit reflex", "5-digit normal"],
        "M":    [2.0,    0.0,    np.nan, np.nan],
        "L":    [np.nan, np.nan, 3.941,  0.329],
        "P":    [4.0,    0.0,    5.424,  6.252],
        "Q":    [np.nan, np.nan, True,   False],
        "XX":   [12.0,   12.0,   16.283, 11.422],
    })

    print(add_geometry_features(test_df))
    