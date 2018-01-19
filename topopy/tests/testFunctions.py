import random
import math
import numpy as np
import scipy
import copy

################################################################################
#Test Functions
################################################################################

################################################################################
## http://math.stackexchange.com/questions/152256/implicit-equation-for-double-torus-genus-2-orientable-surface

def torusPolyGeneratorF(x,n):
  product = 1
  for i in xrange(1,n+1):
    product *= (x-(i-1))*(x-i)
  return product

def torusPolyGeneratorG(x,y,n):
  return (torusPolyGeneratorF(x,n) + y**2)

def doubleTorus(x,y,n=2,r=0.1,sgn=1):
  return sgn*math.sqrt(r**2 - torusPolyGeneratorG(x,y,n)**2)

# def torusPolyGeneratorF(x,y):
#   return (x**2+y**2)**2 - 4*x**2 + y**2

# def doubleTorus(x,y,r=0.1,sgn=1):
#   return sgn*math.sqrt(r**2 - torusPolyGeneratorF(x,y)**2)

def genTorusInputSampleSet(N):
  r=0.175
  n=2
  x = np.zeros(N)
  y = np.zeros(N)
  z = np.zeros(N)

  ## Solved the roots of f(x)=r^2 using wolfram alpha and truncate to make sure
  ## we don't end up with negatives under the radical
  ## Note, changing r will affect these bounds
  minX = -0.073275
  maxX = n-minX

  for i in xrange(N):
    ySgn = (-1)**random.randint(1,2)
    zSgn = (-1)**random.randint(1,2)
    # x[i] = random.uniform(0,n)
    x[i] = random.uniform(minX,maxX)

    ## The distribution below requires less samples, but may be harder to sell
    ## in a paper, as it is clearly non-uniform, whereas the sampling in y-space
    ## makes it look more uniform due to the dependence on x which yields more
    ## samples closer to zero, so we bias towards the more extreme values to
    ## counteract this effect.

    # x[i] = random.betavariate(0.5,0.5)*(maxX-minX) + minX

    fx = torusPolyGeneratorF(x[i],n)
    minY = math.sqrt(max(-r-fx,0))
    maxY = math.sqrt(r-fx)
    # y[i] = ySgn*(random.uniform(minY,maxY))
    y[i] = ySgn*(random.betavariate(0.5,0.5)*(maxY-minY) + minY)
    z[i] = doubleTorus(x[i],y[i],n,r,zSgn)
  return (x,y,z)

## End http://math.stackexchange.com/questions/152256/implicit-equation-for-double-torus-genus-2-orientable-surface
################################################################################

def hinge(_x):
  x = _x[0]
  y = _x[1]
  # x = math.pi/4.*(x)
  # y = math.pi/4.*(y)
  # return math.cos(2*x) - math.cos(y)
  return math.fabs(x-0.55)

def hill(_x):
  x = _x[0]
  y = _x[1]
  return math.exp(- ((x - .55)**2 + (y-.75)**2)/.125) + 0.01*(x+y)

def gerber(_x):
  x = _x[0]
  y = _x[1]
  return   (1./1.) * math.exp(-((x-.25)**2)/0.09) \
         + (1./1.) * math.exp(-((y-.25)**2)/0.09) \
         + (1./3.) * math.exp(-((x-.75)**2)/0.01) \
         + (1./2.) * math.exp(-((y-.75)**2)/0.01)

# def gerber(x,y):
#   return   math.exp(-((x-.25)**2)/0.09) + math.exp(-((y-.25)**2)/0.09) \
#          + math.exp(-((x-.75)**2)/0.01) + math.exp(-((y-.75)**2)/0.01)

def ranjan(_x):
  x = _x[0]
  y = _x[1]
  return x + y + x*y

def goldsteinPrice(_x):
  xa = 4*_x[0]-2;
  ya = 4*_x[1]-2;

  term1 = 1 + (xa+ya+1)**2*(19-14*xa+3*(xa**2)-14*ya+6*xa*ya+3*(ya**2))
  term2 = 30 + (2*xa-3*ya)**2*(18-32*xa+12*(xa**2)+48*ya-36*xa*ya+27*(ya**2))

  return term1*term2

def ridge(_x):
  x = _x[0]
  y = _x[1]
  theta = math.pi/3.
  sigx = .05
  sigy = .04
  a = math.cos(theta)**2/(2*sigx**2) + math.sin(theta)**2/(2*sigy**2)
  b = math.sin(2*theta)/(4*sigx**2)  + math.sin(2*theta)/(4*sigy**2)
  c = math.sin(theta)**2/(2*sigx**2) + math.cos(theta)**2/(2*sigy**2)

  return 0.01*y + 0.5 * (  math.exp(-((x-.75)**2)/0.01)
                         + math.exp(-( (x)**2 + (y-1)**2)/0.1)
                         + math.exp(-( (x)**2 + (y)**2)/0.005)
                         - math.exp(-( a*(x-.25)**2 + 2*b*(x-.25)*(y-.25)
                                       + c*(y-.25)**2)))

def ridge2(_x):
  x = _x[0]
  y = _x[1]
  theta = math.pi/3.
  sigx = .05
  sigy = .04
  a = math.cos(theta)**2/(2*sigx**2) + math.sin(theta)**2/(2*sigy**2)
  b = math.sin(2*theta)/(4*sigx**2)  + math.sin(2*theta)/(4*sigy**2)
  c = math.sin(theta)**2/(2*sigx**2) + math.cos(theta)**2/(2*sigy**2)

  return 0.01*y + 0.5 * (  math.exp(-(((x-.75)**2)/0.01+((y-.5)**2)/0.4))
                         + math.exp(-( (x-.1)**2 + (y-1)**2)/0.1)
                         + math.exp(-( (x-.1)**2 + (y-.1)**2)/0.005)
                         - math.exp(-( a*(x-.3)**2 + 2*b*(x-.3)*(y-.25)
                                       + c*(y-.25)**2)))

def test(_x):
  x = 0.25*_x[0]
  y = 0.25*_x[1]
  return (1./2.) * (  math.exp(-((x-.25)**2)/0.09)
                    + math.exp(-((y-.25)**2)/0.09)
                    + math.exp(-((x-.75)**2)/0.01)
                    + math.exp(-((y-.75)**2)/0.01))

def salomon(_x):
  x = copy.deepcopy(_x)
  if isinstance(x,np.ndarray) and x.ndim > 1:
    d = x.shape[1]
  else:
    d = len(x)

  for i in xrange(d):
    x[i] = 2*x[i]-1

  summand = 0
  for i in xrange(d):
    summand += x[i]**2
  summand = math.sqrt(summand)
  return 1 - math.cos(2*math.pi*summand)+0.1*summand

def salomon2(_x):
  x = copy.deepcopy(_x)
  if isinstance(x,np.ndarray) and x.ndim > 1:
    d = x.shape[1]
  else:
    d = len(x)

  eps = 0
  for i in xrange(d):
    x[i] = 2*x[i]-1
    eps += x[i]*(i+1)

  summand = 0
  for i in xrange(d):
    summand += x[i]**2
  summand = math.sqrt(summand)
  return 1 - math.cos(2*math.pi*summand)+0.1*summand + eps

def strangulation(_x):
  x = _x[0]
  y = _x[1]
  return   (0
         - (0.2) * math.exp(-(math.pow(x-.25,2)+math.pow(y-.25,2))/0.001)
         - (0.2) * math.exp(-(math.pow(x-.25,2)+math.pow(y-.75,2))/0.001)
         - (0.2) * math.exp(-(math.pow(x-.75,2)+math.pow(y-.25,2))/0.001)
         - (0.2) * math.exp(-(math.pow(x-.75,2)+math.pow(y-.75,2))/0.001)
         + (1.) * math.exp(-(math.pow(x-.5,2)+math.pow(y-.5,2))/0.125))
  # return (  0
  #         + (1.) * math.exp(-(math.pow(x-.2,2)+math.pow(y-.2,2))/0.005)
  #         + (1.) * math.exp(-(math.pow(x-.8,2)+math.pow(y-.2,2))/0.005)
  #         - (1.) * math.exp(-(math.pow(x-.5,2)+math.pow(y-.25,2))/0.05)
  #         - (1.) * math.exp(-(math.pow(x-0,2)+math.pow(y-0,2))/0.01)
  #         - (1.) * math.exp(-(math.pow(x-1,2)+math.pow(y-0,2))/0.01)
  #         - (1.) * math.exp(-(math.pow(x-0,2)+math.pow(y-1,2))/0.01)
  #         - (1.) * math.exp(-(math.pow(x-1,2)+math.pow(y-1,2))/0.01)
  #         + (1.) * math.exp(-(math.pow(x-.2,2)+math.pow(y-.8,2))/0.01)
  #         + (1.) * math.exp(-(math.pow(x-.8,2)+math.pow(y-.8,2))/0.01)
  #         + (1.) * math.exp(-(math.pow(x-.5,2)+math.pow(y-.1,2))/0.01)
  #         + (1.) * math.exp(-(math.pow(x-0,2)+math.pow(y-.5,2))/0.01)
  #         + (1.) * math.exp(-(math.pow(x-1,2)+math.pow(y-.5,2))/0.005)
  #         + (1.) * math.exp(-(math.pow(x-.5,2)+math.pow(y-.9,2))/0.01)
  #         + (1.) * math.exp(-(math.pow(x-.45,2)+math.pow(y-.65,2))/0.01)
  #         + (2.) * math.exp(-(math.pow(x-.48,2)+math.pow(y-.52,2))/0.001)
  #         + 1e-2*y)

def himmelblau(_x):
  x = 12*_x[0]-6
  y = 12*_x[1]-6

  return (x**2 + y - 11)**2 + (x+y**2-7)**2

def checkerBoard(_x):
  x = copy.deepcopy(_x)
  if isinstance(x,np.ndarray) and x.ndim > 1:
    d = x.shape[1]
  else:
    d = len(x)

  # for i in xrange(d):
  #   x[i] = 2*x[i] - 1

  periodicTerm = 1
  for i in xrange(d):
    sgn = math.cos(x[i]*math.pi)/abs(math.cos(x[i]*math.pi))
    periodicTerm *= sgn*abs(math.cos(x[i]*math.pi))**(1/7.)
  return periodicTerm

def flatTop(x):
  return checkerBoard(x)*decay(x)

def decay(_x):
  x = copy.deepcopy(_x)
  if isinstance(x,np.ndarray) and x.ndim > 1:
    d = x.shape[1]
  else:
    d = len(x)

  # for i in xrange(d):
  #   x[i] = 2*x[i] - 1

  eps = 0
  for i in xrange(d):
    eps += x[i]*1e-6

  # decayTerm = 0
  # for i in xrange(d):
  #   decayTerm += x[i]**2
  # return math.exp(-decayTerm)
  dist = 0
  for i in xrange(d):
    dist += x[i]**2
  dist = math.sqrt(dist)
  if dist <= 1:
    return 0.1 + (1-dist**3)**3 + eps
  else:
    return 0.1 + eps

def schwefel(_x):
  x = copy.deepcopy(_x)
  if isinstance(x,np.ndarray) and x.ndim > 1:
    d = x.shape[1]
  else:
    d = len(x)
  for i in xrange(d):
    x[i] = x[i]*1000 - 500

  retValue = 418.9829*d
  for i in xrange(d):
    retValue -= x[i]*math.sin(math.sqrt(abs(x[i])))
  return retValue

def ackley(_x):
  x = copy.deepcopy(_x)
  a = 20
  b = 0.2
  c = math.pi*2
  if isinstance(x,np.ndarray) and x.ndim > 1:
    d = x.shape[1]
  else:
    d = len(x)
  for i in xrange(d):
    x[i] = x[i]*3 - 1.5

  summand1 = 0
  summand2 = 0
  for i in xrange(d):
    summand1 += x[i]**2
    summand2 += math.cos(c*x[i])
  eps = 0
  for i in xrange(d):
    eps += 1e-3*x[i]
  return -a*math.exp(-b*math.sqrt(summand1/float(d))) - math.exp(summand2/float(d)) #+eps

def ackley2(_x):
  x = copy.deepcopy(_x)
  a = 20
  b = 0.2
  c = math.pi*2
  if isinstance(x,np.ndarray) and x.ndim > 1:
    d = x.shape[1]
  else:
    d = len(x)
  for i in xrange(d):
    x[i] = x[i]*1.5

  summand1 = 0
  summand2 = 0
  for i in xrange(d):
    summand1 += x[i]**2
    summand2 += math.cos(c*x[i])
  eps = 0
  for i in xrange(d):
    eps += 1e-3*(i+1)*x[i]
  return -a*math.exp(-b*math.sqrt(summand1/float(d))) - math.exp(summand2/float(d)) + eps

def torus(u,v):
  c = 1
  a = 0.25
  x = (c+a*math.cos(v))*math.cos(u)
  y = (c+a*math.cos(v))*math.sin(u)
  z = a*math.sin(v)
  return [x,y,z]

def shekel(_x):
  x = copy.deepcopy(_x)
  if isinstance(x,np.ndarray) and x.ndim > 1:
    d = x.shape[1]
  else:
    d = len(x)

  for i in xrange(d):
    x[i] = 10*x[i]

  m = 4
  a = np.zeros((m,d))
  c = np.ones(m)

  # i = 0 (center of domain)
  c[0] = 0.25
  for j in xrange(d):
    a[0,j] = 5
  # i = 1 (upper corner)
  c[1] = 0.5
  for j in xrange(d):
    a[1,j] = 10
  # i = 1 (odd low, even high)
  c[2] = 0.75
  for j in xrange(d):
    if j % 2 == 0:
      a[2,j] = 1
    else:
      a[2,j] = 9
  # i = 3 (odd high, even low)
  c[3] = 1
  for j in xrange(d):
    if j % 2 == 0:
      a[3,j] = 9
    else:
      a[3,j] = 1

  summand = 0
  for i in xrange(m):
    subSummand = c[i]
    for j in xrange(d):
      subSummand += (x[j]-a[i,j])**2
    summand += (subSummand)**-1
  return summand

def rosenbrock(_x):
  x = copy.deepcopy(_x)
  if isinstance(x,np.ndarray) and x.ndim > 1:
    d = x.shape[1]
  else:
    d = len(x)
  for i in xrange(d):
    x[i] = 4.8*x[i] - 2.4

  return scipy.optimize.rosen(x)

def rosenbrock2(_x):
  x = copy.deepcopy(_x)
  if isinstance(x,np.ndarray) and x.ndim > 1:
    d = x.shape[1]
  else:
    d = len(x)
  for i in xrange(d):
    x[i] = 4.8*x[i]-2.4

  eps = 0
  return scipy.optimize.rosen(x) + eps

def gerber2(_x):
  x = _x[0]
  y = _x[1]
  return   (1./2.) * math.exp(-((x-.25)**2)/0.09) \
         + (3./4.) * math.exp(-((x-.75)**2)/0.01) \
         +           math.exp(-((y-.75)**2)/0.01)

testFunctions = {}
## Trivial
testFunctions['ranjan'] = ranjan
testFunctions['test'] = test
testFunctions['decay'] = decay

## Working
testFunctions['ackley'] = ackley
testFunctions['flatTop'] = flatTop
testFunctions['gerber'] = gerber
testFunctions['goldsteinPrice'] = goldsteinPrice
testFunctions['hill'] = hill
testFunctions['himmelblau'] = himmelblau
testFunctions['hinge'] = hinge
testFunctions['ridge'] = ridge
testFunctions['ridge2'] = ridge2
testFunctions['rosenbrock'] = rosenbrock  ##Crossing dendrogram lines
testFunctions['salomon'] = salomon
testFunctions['schwefel'] = schwefel  ## Probably the most genuinely complicated
testFunctions['shekel'] = shekel
testFunctions['strangulation'] = strangulation
# testFunctions['torus'] = torus

## Under test
testFunctions['rosenbrock2'] = rosenbrock2
testFunctions['salomon2'] = salomon2

## Broken
## NONE! Woohoo!
