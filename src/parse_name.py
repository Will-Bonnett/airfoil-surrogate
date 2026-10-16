import pytest

def parse_name(name: str) -> dict:
    arr = name.split("_")

    if arr[0] != "airFoil2D" or arr[1] != "SST":
        raise ValueError("Malformed Name")

    return {
        "inletVelocity":float (arr[2]),
        "angleOfAttack":float(arr[3]),
    } | parse_naca(arr[4:])


def parse_naca(naca: str) -> dict:
    if len(naca) == 3:
        return {
            "M" : float(naca[0]), # Maximum Camber (how curved the airfoil is)
            "P" : float(naca[1]), # Where along the meximum is of chord
            "XX" : float(naca[2]) # Maximum thickness as a % of chord
        }
    elif len(naca) == 4:
        return {
            "L": float(naca[0]), # design lift coefficient
            "P": float(naca[1]), # position of meximum camber
            "Q": True if naca[2]=="1.0" else False, # flag, "0" normal, "1" reflex camber
            "XX" : float(naca[3]) # thickness
        }


if __name__=="__main__":
    print(parse_name("airFoil2D_SST_43.597_5.932_3.511_3.1_1.0_18.252"))


