from occsfrd import wick

def texify(equations, filename):
    with open(filename + ".tex", "w") as f:
        f.write("\documentclass{minimal}")
        f.write("\n\n")
        f.write("\\begin{document}")
        # for equation in equations:
        #     f.write("\n")
        #     texifySingleEquation(equation, f)
        #     f.write("\n")
        texifyCollectionOfEquations(equations, f)
        f.write("\n")
        f.write("\end{document}")

def texifyCollectionOfEquations(equations, file):
    for equation in equations:
        file.write("\n")
        if isinstance(equation, wick.tensor.TensorSum):
            texifySingleEquation(equation, file)
        elif isinstance(equation, list) or isinstance(equation, tuple):
            texifyCollectionOfEquations(equation, file)
        else:
            return NotImplemented
        file.write("\n")

def texifySingleEquation(equation, file):
    file.write("$$")
    file.write(equation.__str__())
    file.write("$$")