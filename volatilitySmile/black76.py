import math
from test_plotImpliedVol import *
import matplotlib.pyplot as plt

'''
Toy project script to construct volatility smile/ surface using black76 (by: zhou xinzi/ zhou0184@e.ntu.edu.sg)
'''

RISK_FREE_RATE = 0.01 # this is the continuous risk-free rate
EPSILON = 1e-6 

START_LOWER = 0.01      # starting lower bound
START_UPPER = 2         # starting upper bound


def black76(isCall, r, F, K, sigma, T):
    '''
    https://en.wikipedia.org/wiki/Black_model

    input: 
    isCall  - true for call/ false for put
    r       - risk-free rate 
    F       - futures price [where ln(F(t)) ~ norm(µ,σ) // σ is constant]
    K       - strike
    sigma   - σ == constant volatility
    T       - time to maturity

    output: 
    callPrice       - e**(-rT) * (F*N(d1) - K*N(d2))
    putPrice        - e**(-rT) * (K*N(-d2) - F*N(-d1))

    where
    d_1             - ( ln(F/K) + 0.5 * (σ**2) * T ) / ( σ * √(T) ) 
    d_2             - d_1  -  σ * √(T),

    '''
    scaledVol = sigma * (T ** 0.5) 
    d1 = (math.log( F / K ) + 0.5 * ( sigma ** 2 ) * T) / scaledVol
    d2 = d1 - scaledVol

    if isCall: 
        return math.exp(-r*T) * (F * N(d1) - K * N(d2))
    else: 
        return math.exp(-r*T) * (K * N(-d2) - F * N(-d1))


def N(x):
    ''' Computes cumulative normal distribution, using built-in math library's standard error function '''
    return (1.0 + math.erf(x/ math.sqrt(2.0))) / 2.0


def bisection(isCall, r, F, K, T, Premium, lower, upper, lowerError = None): 
    '''
    bisection search for an unknown sigma
    
    implemented following methodology: https://investexcel.net/calculate-implied-volatility-with-the-bisection-method/
    '''

    # cleanse data points with negative time value
    timeValue = Premium - max(0, (1 if isCall else -1.) * (F - K))
    if timeValue < -EPSILON : return None 
    
    # while black76(isCall, r, F, K, upper, T) - Premium < 0: 
    #     upper *= 2
    
    if lowerError is None: 
        lowerError = black76(isCall, r, F, K, lower, T) - Premium

    if abs(lowerError) < EPSILON: 
        return lower

    mid = (lower + upper) / 2
    midError = black76(isCall, r, F, K, mid, T) - Premium

    if abs(midError) < EPSILON: 
        
        return mid
    if lowerError * midError < 0: # opp sign, so search in between
        return bisection(isCall, r, F, K, T, Premium, lower, mid, lowerError)
    else:
        return bisection(isCall, r, F, K, T, Premium, mid, upper, midError)

def prepData(strikes, types, optionPrices, futurePrices, time_to_expiry):
    ''' 
    input (individual lists/ assumes equilength data): 
    [strike], [type], [optionPrice], [futurePrices], [time_to_expiry]

    output (list of tuples):
    [(isCall, r, F, K, T, Premium)...]
    '''

    l = len(strikes)
    return list(zip(
        [x == "C" for x in types],
        [RISK_FREE_RATE] * l, 
        futurePrices,
        strikes,
        time_to_expiry,
        optionPrices, 
        [START_LOWER] * l, 
        [START_UPPER] * l))


def splitSeries(strikes, vols, ttms):
    '''split series according to time_to_expiry'''
    retDict = {}
    for i in range(len(strikes)):
        if retDict.get(ttms[i]) is None:
            retDict[ttms[i]] = ([strikes[i]], [vols[i]])
        else:
            retDict[ttms[i]][0].append(strikes[i])
            retDict[ttms[i]][1].append(vols[i])
    
    return retDict
    

def plotImpliedVol(): 
    '''
    prepares raw data into required format, 
    create and plot volatility smiles 
    '''
    ########## data prep #########
    inputs = prepData(STRIKE, TYPE, OPTION_PRICE, FUTURE_PRICE, TIME_TO_EXPIRY)
    strikes = []
    iv = []
    ttm = []
    
    ########## do analysis of IV #########
    for i in range(len(STRIKE)):
        ans = bisection(*(inputs[i]))

        if ans is not None:
            strikes.append(STRIKE[i])
            iv.append(ans)
            ttm.append(TIME_TO_EXPIRY[i])


    splitData = splitSeries(strikes, iv, ttm)

    ########## start plotting #########

    fig, axs = plt.subplots(len(splitData), sharex=True)
    fig.set_size_inches(6.0, 12.5)
    fig.suptitle("Volatility smiles")

    i = 0
    for k in splitData.keys():
        axs[i].scatter(splitData[k][0], splitData[k][1])
        axs[i].set_title('Time to expiry = ' + str(k), y=1.0, pad=-14, fontsize=8.0, loc='right')
        i += 1
    
    for ax in axs:
        ax.label_outer()

    plt.savefig("smile.png")

    # finally we plot the 3d 
    newFig = plt.figure(2)
    ax3d = newFig.add_subplot(projection='3d')
    ax3d.scatter(strikes, iv, ttm, color='r')
    ax3d.set_xlabel('Strike Price')
    ax3d.set_ylabel('Implied Volatility')
    ax3d.set_zlabel('Time to expiry')
    

    plt.show()


plotImpliedVol()


