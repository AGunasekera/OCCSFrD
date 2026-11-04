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

def test_mulBasic(basic_cre_p0a, basic_ann_p0a):
    assert 2. * basic_cre_p0a == operator.OperatorProduct([basic_cre_p0a], 2.)
    assert basic_cre_p0a * 2. == operator.OperatorProduct([basic_cre_p0a], 2.)
    assert (basic_cre_p0a * basic_ann_p0a) == operator.OperatorProduct([basic_cre_p0a, basic_ann_p0a])

def test_isProportional(basic_cre_p0a, basic_ann_p0a, basic_cre_p1a, basic_ann_p1a):
    prod1 = operator.OperatorProduct([basic_cre_p0a, basic_ann_p0a], 1.)
    prod2 = operator.OperatorProduct([basic_cre_p0a, basic_ann_p0a], 2.)
    prod3 = operator.OperatorProduct([basic_cre_p0a, basic_ann_p1a], 1.)

    assert prod1.isProportional(prod2)
    assert not prod1.isProportional(prod3)
            
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

def test_collectSummandList(basic_cre_p0a, basic_ann_p0a, basic_cre_p1a, basic_ann_p1a):
    prod1 = operator.OperatorProduct([basic_cre_p0a, basic_ann_p0a], 1.)
    prod2 = operator.OperatorProduct([basic_cre_p0a, basic_ann_p0a], 2.)
    sum = operator.OperatorSum([operator.OperatorProduct([basic_cre_p0a, basic_ann_p1a], 3.)])

    assert operator.OperatorSum([prod1, prod2]).collectSummandList() == sum

def test_conjugateSum(basic_cre_p0a, basic_ann_p0a, basic_cre_p1a, basic_ann_p1a):
    sum = (basic_cre_p0a * basic_ann_p0a) + (basic_cre_p1a * basic_ann_p1a)
    sum2creation = (basic_cre_p0a * basic_cre_p0a) + (basic_cre_p1a * basic_cre_p1a)
    sum2annihilation = (basic_ann_p0a * basic_ann_p0a) + (basic_ann_p1a * basic_ann_p1a)
    
    assert sum.conjugate() == sum
    assert sum2creation.conjugate() == sum2annihilation
            
def test_normalOrder(basic_cre_p0a, basic_ann_p0a, basic_cre_p1a, basic_ann_p1a):
    prod1 = operator.OperatorProduct([basic_cre_p0a, basic_ann_p0a], 1.)
    prod2 = operator.OperatorProduct([basic_ann_p0a, basic_cre_p0a], 1.)
    prod3 = operator.OperatorProduct([basic_cre_p0a, basic_ann_p0a], -1.)
    
    assert operator.normalOrder(prod1) == prod1
    assert operator.normalOrder(prod2) == prod3

def test_excitation(index_p0, index_p1, basic_cre_p0a, basic_ann_p1a):
    ex = operator.excitation([index_p0], [index_p1], [True])
    assert ex == operator.OperatorProduct([basic_cre_p0a, basic_ann_p1a])

def test_spinFreeExcitation(index_p0, index_p1, basic_cre_p0a, basic_ann_p1a, basic_cre_p0b, basic_ann_p1b):
    ex = operator.spinFreeExcitation([index_p0], [index_p1])
    assert ex == operator.OperatorSum([operator.OperatorProduct([basic_cre_p0a, basic_ann_p1a]), operator.OperatorProduct([basic_cre_p0b, basic_ann_p1b])])