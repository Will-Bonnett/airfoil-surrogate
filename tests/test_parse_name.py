import pytest
from parse_name import parse_name

def test_5digit_name():
    name = "airFoil2D_SST_43.597_5.932_3.511_3.1_1.0_18.252"
    dic = parse_name(name)

    assert dic["inletVelocity"] == 43.597
    assert dic["angleOfAttack"] == 5.932
    assert dic["L"] == 3.511
    assert dic["P"] == 3.1
    assert dic["Q"] == True
    assert dic["XX"] == 18.252


def test_4digit_name():
    name = "airFoil2D_SST_43.597_5.932_0_0_12"
    dic = parse_name(name)

    assert dic["inletVelocity"] == 43.597
    assert dic["angleOfAttack"] == 5.932
    assert dic["M"] == 0.0
    assert dic["P"] == 0.0
    assert dic["XX"] == 12.0


def test_malformed1_name():
    name = "Foil_SST_43.597_5.932_0_0_12"
    with pytest.raises(ValueError):
        parse_name(name)

def test_malformed2_name():
    name = "airFoil2D_STT_43.597_5.932_0_0_12"
    with pytest.raises(ValueError):
        parse_name(name)
    