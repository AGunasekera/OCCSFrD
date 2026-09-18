from occsfrd.wick import index

def test__hash__(first, second, different):
    assert hash(first) == hash(second)
    assert hash(first) != hash(different)

def test__eq__(first, second, different):
    assert first == second
    assert first != different

def test__str__(first, different):
    assert str(first) == "i"
    assert str(different) == "a"

def test_contractedCopy(copied, general):
    assert copied.contractedFrom is general

def test__copy__(copied):
    assert copied.name == "a"