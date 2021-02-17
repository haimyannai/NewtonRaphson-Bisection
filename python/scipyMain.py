from scipy import optimize
import numpy as np
import time


def bisection(fx, dfx, section, epsilon, iterations):
    # There is a root
    a = section[0]
    b = section[1]
    dfx2 = np.polyder(dfx)
    for i in range(1, iterations):
        if a < b and fx(a) * fx(b) < 0:
            xMid = (a + b) / 2
            fxMid = fx(xMid)
            if abs(fxMid) > epsilon:
                xMid = optimize.newton(fx, xMid, fprime=dfx, args=(), maxiter=iterations, fprime2=dfx2)
                fxMid = fx(xMid)
                if xMid is not None and fxMid < epsilon:
                    return xMid
                if fxMid < 0:
                    a = xMid
                else:
                    b = xMid
            else:
                return xMid
        return None


def checkNewRoot(newRoot, rootLst, rootDiff=0.5):
    return len(list(filter((lambda root: abs(newRoot - root) < rootDiff), rootLst))) == 0


def calculateRoots(cofficieArr, epsilon=0.001):
    guessRootsLst = list()
    rootsLst = list()
    rootDiff = 0.001
    maxIterations = 10  # maximum iterations to search for roots between a and b
    func = np.seterr(all='ignore')  # inorder to refuse overflow error
    func = np.poly1d(cofficieArr)  # change cofficient array to function: [-1,0,3,3]---> -x^3+3x+3
    dfunc = np.polyder(func)

    # FIND GUESSES OF ROOTS BY BISECTION METHOD
    section = np.arange(-8, 8, 0.01).tolist()
    sectionT = []
    for i in range(len(section) - 1):
        j = i + 1
        sectionT.append((section[i], section[j]))
    for sec in sectionT:
        if abs(func(sec[0])) > epsilon:
            checkNextGuess = bisection(func, dfunc, sec, epsilon, maxIterations)
        else:
            checkNextGuess = sec[0]
        if checkNextGuess is not None and checkNewRoot(checkNextGuess, rootsLst, rootDiff):
            rootsLst.append(checkNextGuess)

    return list(filter(None, list(set(rootsLst))))


def main():
    # input
    with open('data_poly.txt') as myFile:
        text = myFile.read()
    epsilon = 0.0000001
    sumTime = 0

    coffieciantLst = np.array(text.split(",")).astype(np.float)  # creating list of coffieciant
    startTime = time.time()
    roots = calculateRoots(coffieciantLst, epsilon)
    finishTime = time.time()
    sumTime += finishTime - startTime

    roots = sorted(set(roots))
    print(len(roots), " roots found:")
    i = 1
    for root in roots:
        print("root #", i, ": ", root)
        i += 1
    print("total time:", sumTime)


main()
