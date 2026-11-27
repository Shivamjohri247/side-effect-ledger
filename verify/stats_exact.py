import math
def binom_cdf(k,n,p): return sum(math.comb(n,i)*p**i*(1-p)**(n-i) for i in range(k+1))
def cp(x,n,a=.05):
    def bis(f,lo=0.,hi=1.):
        for _ in range(100):
            m=(lo+hi)/2
            if f(m): lo=m
            else: hi=m
        return (lo+hi)/2
    lo=0. if x==0 else bis(lambda p: 1-binom_cdf(x-1,n,p) < a/2)
    hi=1. if x==n else bis(lambda p: binom_cdf(x,n,p) > a/2)
    return lo,hi
def wilson(x,n,z=1.959964):
    p=x/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return max(0,c-h),min(1,c+h)
def sign(a,b):
    wa=sum(x>y for x,y in zip(a,b)); wb=sum(y>x for x,y in zip(a,b)); n=wa+wb
    if n==0: return wa,wb,1.
    return wa,wb,min(1.,2*sum(math.comb(n,i) for i in range(min(wa,wb)+1))/2**n)
def holm(ps):
    o=sorted(range(len(ps)),key=lambda i:ps[i]); m=len(ps); adj=[0]*m; run=0
    for r,i in enumerate(o): run=max(run,min(1,(m-r)*ps[i])); adj[i]=run
    return adj
assert abs(cp(0,27)[1]-0.1277)<1e-3 and abs(sign([1]*6,[0]*6)[2]-0.03125)<1e-9
