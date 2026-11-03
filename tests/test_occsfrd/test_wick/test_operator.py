from copy import copy
from occsfrd.wick import index, operator

def test_conjugateBasic(basic_cre_p0a, basic_ann_p0a):
    assert basic_cre_p0a.conjugate() == basic_ann_p0a
    assert basic_ann_p0a.conjugate() == basic_cre_p0a

def test_copyBasic(basic_cre_p0a):
    assert copy(basic_cre_p0a) == basic_cre_p0a
    assert not (copy(basic_cre_p0a) is basic_cre_p0a)

def test_strBasic(basic_cre_p0a):
    assert str(basic_cre_p0a) == "a^{p0\\alpha}"

def test_mulBasic(basic_cre_p0a, basic_ann_p0a, index_p0):
    assert 2. * basic_cre_p0a == operator.OperatorProduct([basic_cre_p0a], 2.)
    assert basic_cre_p0a * 2. == operator.OperatorProduct([basic_cre_p0a], 2.)
    assert (basic_cre_p0a * basic_ann_p0a) == operator.OperatorProduct([basic_cre_p0a, basic_ann_p0a])

def test_isProportional():
    assert True
            
def test_checkNilpotency(basic_cre_p0a, basic_ann_p0a):
    assert (basic_cre_p0a * basic_ann_p0a).checkNilpotency()
    assert not (basic_cre_p0a * basic_cre_p0a).checkNilpotency()

def test_conjugateProduct(basic_cre_p0a, basic_ann_p0a, basic_cre_p1a, basic_ann_p1a):
    assert (basic_cre_p0a * basic_ann_p0a).conjugate() == basic_cre_p0a * basic_ann_p0a
    assert (basic_cre_p0a * basic_cre_p1a).conjugate() == basic_ann_p1a * basic_ann_p0a

def test_copyProduct(basic_cre_p0a, basic_ann_p0a):
    prod = basic_cre_p0a * basic_ann_p0a
    assert copy(prod) == prod
    assert not (copy(prod) is prod)

def test_productEquality():
    g0, g1 = index.Index("g_{0}", False), index.Index("g_{1}", False)
    assert operator.BasicOperator(g0, True, False) * operator.BasicOperator(g1, False, False) == operator.BasicOperator(g0, True, False) * operator.BasicOperator(g1, False, False)
    assert not operator.BasicOperator(g0, True, False) * operator.BasicOperator(g1, False, False) == (operator.BasicOperator(g0, True, True) * operator.BasicOperator(g1, False, True))

def test_collectSummandList():
    assert True

def test_conjugateSum():
    assert True
            
def test_normalOrder():
    assert True

def test_excitation():
    assert True

def test_spinFreeExcitation():
    assert True