import numpy as np
import time

# Newton Raphson method
def NR(fx, dfx, epsilon, iterations, xn):
    xnDev = dfx(xn) # xnDev = f'(xn)
    if xnDev is not 0:
        xn1 = xn - fx(xn) / xnDev # take the next x point newton raphson
        t = abs(fx(xn1))
        if t < epsilon:
            return xn1
    for i in range(2, iterations):
        xnDev = dfx(xn1)
        if xnDev is not 0:
            xn1 = xn1 - fx(xn1) / xnDev
            t = abs(fx(xn1))
            if t < epsilon:
                return xn1
    return xn1

# Bisection method
def bisection(fx, dfx, section, epsilon, iterations):
    # There is a root
    a = section[0]
    b = section[1]
    for i in range(1, iterations):
        if a < b and fx(a) * fx(b) < 0: # there is a root bisection
            xMid = (a + b) / 2
            fxMid = fx(xMid)
            if abs(fxMid) > epsilon:
                # sendig xMid to Newton Raphson
                xMid = NR(fx=fx, dfx=dfx, epsilon=epsilon, iterations=iterations, xn=xMid)
                t = fx(xMid)
                if xMid is not None and abs(fx(xMid)) < epsilon:
                    return xMid
                if t < 0:
                    #change section size by xMid
                    a = xMid
                else:
                    b = xMid
            else:
                return xMid
        return None

# checking if we found a new root -> a new root must be |newRoot otherRoots|<rootDiff
def checkNewRoot(newRoot, rootLst, rootDiff = 0.5):
    return len(list(filter((lambda root: abs(newRoot - root) < rootDiff), rootLst))) == 0


def calculateRoots(cofficieArr, epsilon=0.001):
    rootsLst = list()
    rootDiff = 0.001
    maxIterations = 10  # maximum iterations to search for roots between a and b
    func = np.seterr(all='ignore')  # inorder to refuse overflow error
    func = np.poly1d(cofficieArr)  # change cofficient array to function: [-1,0,3,3]---> -x^3+3x+3
    dfunc = np.polyder(func) # f'(x)
    # FIND GUESSES OF ROOTS BY BISECTION METHOD
    section = np.arange(-8, 8, 0.01).tolist() # [a,b]
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
    #allTimeForFuncs.append(finishTime - startTime)
    sumTime += finishTime - startTime
    roots = sorted(set(roots))
    print(len(roots), " roots found:")
    i = 1
    for root in roots:
        print("root #", i, ": ", root)
        i += 1
    print("total time:", sumTime)


main()
