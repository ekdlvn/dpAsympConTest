# Functions for Differentially Private Goodness of fit test and independence Testing
import numpy as np
#######################################################
# Set of function to conduct goodness-of-fit tests    #
#                                                     #
# input:                                              #
#  - [required] X: original table (vector)            #
#  - [required] mu (sigma): privacy level or noise    #
#                                                     #
# output:                                             #
#  - U: perturbed table (vector)                      #
#######################################################
# def chk():
#     print("Newly Compiled Python File!")

def Sig(mu: float, n: int, sens: float) -> float:
    """ Return standard deviation of noise when privacy budget and sample size is given.

    Args:
        mu (float): Privacy Budget
        n (int): Sample size
        sens (float): Sensitivity

    Returns:
        float: Standard deviation of gaussian noise
    """
    return sens/mu/np.sqrt(n)

missing = object()
# def noisyTable(x, mu=None, sens=None, sig=None) -> np.ndarray:
#     """Function to generate noisy data 
#     input:                          
#     - x: original data             
#     - mu: privacy parameter        
#     - sens: sensitivity            
                                    
#     output:                         
#     - noisy data                   
#     """
#     if x is missing: print('There is no default values for the function.')
#     n = sum(x)
#     d = len(x)
#     if (mu is None) and (sig is None): print('At least one value is required for mu or sig!')
#     if (sig is None) and (sens is None): print('sens is required when only mu is provided.')
#     if sig is None: sig = Sig(mu, sum(x), sens)
#     noisyTab = x + np.random.normal(0, sig*np.sqrt(n), d)
#     return noisyTab

def noisyTable(x: np.ndarray, mu: float=None, sens: float=None, sig: float=None) -> np.ndarray:
    """Return noisy table perturbed satisfying GDP when privacy budget and sensitivity or standard deviation of noise are/is given.

    Args:
        x (np.ndarray): Input vector or array 
        mu (float, optional): Privacy parameter for GDP. Defaults to None.
        sens (float, optional): Sensitivity to compute sigma when mu is given. Defaults to None.
        sig (float, optional): Standard deviation of noise. If mu and sens is given, sig can be computed. Defaults to None.

    Returns:
        np.ndarray: Noisy table satisfying GDP
    """
    if x is missing: print('There is no default values for the function.')
    n = sum(x)
    d = len(x)
    if sig is None:
        if mu is None: print('At least one value is required for mu or sig!')
        if sens is None: print('sens is required when only mu is provided.')
        sig = Sig(mu, sum(x), sens)
    noisyTab = x + np.random.normal(0, sig*np.sqrt(n), d)
    return noisyTab
    
# #################################################################
# #                                                               #
# # Set of functions to compute Sigma for Independence testing    #
# #                                                               #
# # input:                                                        #
# #  - [required] pi_1: row marginal probabilities                #
# #  - [required] pi_2: column marginal probabilities             #
# #  - [required] sigma: st. dev. of noise                        #
# #  - [required] type: type of test statistics                   #
# #                                                               #
# # output:                                                       #
# #  - Sigma: Estimated covariance matrix                         #
# #################################################################

def ones(d: int) -> np.ndarray:
    """ Return length d one vector.

    Args:
        d (int): Number of bins

    Returns:
        np.ndarray: Size d vector with elements 1
    """
    return np.repeat(1, d)

####################
# function : nabla #
####################
def nabla(p_y: np.ndarray, p_x: np.ndarray) -> np.ndarray:
    """ Return nabla matrix when marginal probabilities are given.

    Args:
        p_y (np.ndarray): Marginal probabilities for columns
        p_x (np.ndarray): Marginal probabilities for rows

    Returns:
        np.ndarray: Matrix for 
    """
    c = len(p_y)
    r = len(p_x)
    # change to column vector ; python treat array as a row vector
    p_y.shape = (c,1)
    p_x.shape = (r,1)

    part_py = np.kron(p_y, np.vstack((np.identity(r-1), -ones(r-1))))
    part_px = np.kron(np.vstack((np.identity(c-1), -ones(c-1))), p_x)
    
    nab = np.column_stack((part_py, part_px))
    return nab


# ####################
# # function : prj.A #
# ####################
# # prj.A required to compute covariance of limiting distribution based on pi(probabilities of cells)
# prj.A = function(pi_1, pi_2){
#   ## input
#   # pi_1: marginal probabilities of row variables
#   # pi_2: marginal probabilities of column variables
  
#   pi = kronecker(pi_2, pi_1)
  
#   r = length(pi_1)
#   c = length(pi_2)
  
#   Jc = matrix(1, c, c)
#   Jr = matrix(1, r, r)
  
#   D.half.pi = diag(sqrt(pi))
#   D.half.inv = diag(1/sqrt(pi))
#   D_pi1 = diag(pi_1)
#   D_pi2 = diag(pi_2)
  
#   # nab = cbind(kronecker(pi_2, rbind(diag(r-1), -t(one(r-1)) ) ),
#   #             kronecker(rbind(diag(c-1), -t(one(c-1))), pi_1)) 
  
#   nab = nabla(pi_2, pi_1)
#   A = D.half.inv %*% nab
#   Q = qr.Q(qr(A))
#   prj.A = tcrossprod(Q) # Q%*%t(Q)
#   # prj.A = A%*%solve(t(A)%*%A)%*%t(A)
#   # prj.A = ifelse(prj.A > 1e-14, prj.A, 0) # this will change the results a lot # we need to fix the eigen values at the end of the procedures.
  
#   return(list(A=prj.A, pi=pi, r=r, c=c, Jc=Jc, Jr=Jr, D.half.pi = D.half.pi, D.half.inv = D.half.inv, D_pi1 = D_pi1, D_pi2=D_pi2))
# }

# ####################
# # function cov.dat #
# ####################
# # covariance consists of data covariance for the limiting distribution from multinomial distributions

# covDat = function(pi_1, pi_2){
#   prj.A = prj.A(pi_1, pi_2)
#   A = prj.A$A; pi = prj.A$pi; r= prj.A$r; c=prj.A$c
#   d.x = length(pi_1); d.y = length(pi_2)
#   # cov.dat = diag(d.x*d.y) - outer(sqrt(pi), sqrt(pi)) - A
#   cov.dat = diag(c*r) - outer(sqrt(pi), sqrt(pi)) - A
#   return(list(cov.dat = cov.dat, pi=pi, r=r, c=c, prjA = prj.A))
# }

# #####################
# # function : SigInG #
# #####################
# # elements consisting covariance matrix of the given distribution
# SigInG = function(pi_1, pi_2, sig, type="I"){
#   if(!type%in%c("I", "n", "G", "a", "In")){
#     stop("Input type is wrong")
#   }
#   # tryCatch(
#   #   error=function(e){
#   #     cat("pi_1:", pi_1, "\n")
#   #     cat("pi_2:", pi_2, "\n")
#   #   }
#   # )
#   cov = covDat(pi_1, pi_2)
#   cov.dat = cov$'cov.dat'
#   pi = cov$pi ; r = cov$r; c=cov$c; prjA = cov$prjA
#   Jr=prjA$Jr; Jc = prjA$Jc
#   D_pi1 = prjA$D_pi1; D_pi2 = prjA$D_pi2; D.half.inv = prjA$D.half.inv; D.half.pi = prjA$D.half.pi
  
#   A3 = D_pi2%*%Jc
#   A4 = D_pi1%*%Jr
  
#   # A1 = (diag(c)-D_pi2%*%Jc)
#   # A2 = (diag(r)-D_pi1%*%Jr)
  
#   A1 = diag(c)-A3
#   A2 = diag(r)-A4
  
#   Sig.sig.pi = sig^2 * kronecker(tcrossprod(A1), tcrossprod(A2))
#   # Sig.sig.pi = ifelse(Sig.sig.pi < 1e-15, 0, Sig.sig.pi)
  
  
#   if(type%in%c("n", "G", "a", "In")){
#     # A3 = D_pi2%*%Jc
#     # A4 = D_pi1%*%Jr
    
#     Sig.sig.pi.n = Sig.sig.pi + sig^2 * (kronecker(tcrossprod(A1, A3), tcrossprod(A2, A4)) # (A1%*%t(A3)), (A2%*%t(A4))) 
#                                          + kronecker(tcrossprod(A3, A1), tcrossprod(A4, A2))  #(A3%*%t(A1)), (A4%*%t(A2)))
#                                          + kronecker(tcrossprod(A3), tcrossprod(A4)) #(A3%*%t(A3), A4%*%t(A4))
#     )
#     # Sig.sig.pi.n = ifelse(Sig.sig.pi < 1e-15, 0, Sig.sig.pi.n)
#   }
  
#   if(type%in%c("G", "a")){
#     # A5 = kronecker(one(c), matrix(pi_1,nrow=length(pi_1)))%*%t(one(r*c))/c
#     # A5 = kronecker(Jc, D_pi1%*%Jr)/c
#     # A6 = kronecker(Jr, D_pi2%*%Jc)/r
#     A5 = kronecker(Jc, A4)/c
#     A6 = kronecker(Jr, A3)/r  
#     Sig.sig.pi.g = Sig.sig.pi + sig^2 * (-kronecker(A3%*%t(A1), A4%*%t(A2)) # sig^2 * kronecker(tcrossprod(A1), tcrossprod(A2)) = Sig.sig.pi
#                                          +A5%*%kronecker(t(A1), t(A2))
#                                          +A6%*%kronecker(t(A1), t(A2))
#                                          - kronecker(A1%*%t(A3), A2%*%t(A4))
#                                          + kronecker(A3%*%t(A3), A4%*%t(A4))
#                                          - A5%*% kronecker(t(A3), t(A4))
#                                          - A6%*% kronecker(t(A3), t(A4))
#                                          + kronecker(A1, A2)%*%t(A5)
#                                          - kronecker(A3, A4)%*%t(A5)
#                                          + A5%*%t(A5)
#                                          + A6%*%t(A5)
#                                          + kronecker(A1, A2)%*%t(A6)
#                                          - kronecker(A3, A4)%*%t(A6)
#                                          + A5%*%t(A6)
#                                          + A6%*%t(A6)
#     )
    
#     # Sig.sig.pi.g = ifelse(Sig.sig.pi.g < 1e-15, 0, Sig.sig.pi)
#   }
  
#   Sig.I = cov.dat + D.half.inv %*% Sig.sig.pi %*% D.half.inv
#   if(type=="I") return(Sig.I)
  
#   Sig.n = cov.dat + D.half.inv %*% Sig.sig.pi.n %*% D.half.inv
#   if(type=="n") return(Sig.n)
  
#   if(type=="In") return(list(Sig.I = Sig.I, Sig.n = Sig.n))
  
#   Sig.G = cov.dat + D.half.inv %*% Sig.sig.pi.g %*% D.half.inv
#   if(type=="G") return(Sig.G)
  
#   if(type=="a") return(list(Sig.I = Sig.I, Sig.n = Sig.n, Sig.G = Sig.G))
# }

# #######################################################
# #                                                     #
# # Set of functions to conduct independence testing    #
# #                                                     #
# # input:                                              #
# #  - [required] U: noisy table (vector)               #
# #  - [required] mu (sigma): privacy level or noise    #
# #  - [required] r: the number of rows                 #
# #  - [required] c: the number of column               #
# #  - [required] statType: statType of test statistic  #
# #  - [required] adjType: adjustment of estimates      #
# #  - [required] sens: sensitivity of table            #
# #  - [optional] n: sample size                        #
# #                                                     #
# # output:                                             #
# #  - Sigma: Estimated covariance matrix               #
# #  - lambda: eigenvalues of Sigma                     #
# #  - hat.pi: estimated marginal probabilities         #
# #  - quantile of null dist: critical value            #
# #  - testStat: test statistic                         #
# #  - p.val: p-value                                   #
# #  - plot of null distribution                        #
# #######################################################

# #######################################################################################
# # optional:                                                                           #
# #  - when sample size is given testStat should be determined based on the sample size #
# #  - for independence testing, the statistics would be two                             #
# #######################################################################################

# # Function : subSmall #
# # take abs of estimates #

# subSmall = function(x){
#   l = sum(x<=0)
#   k = sum(x[x>0])/(100-1)
#   ifelse(x <0, k, x) %>% {./sum(.)}
# }

# #         Function : testStat           #
# # compute test statistic and estimates  #

# testStat = function(u, r, c, n, statType){
#   if(!is.matrix(u)){
#     if(missing(r) && missing(c)) {stop("The number of rows or columns is required.")}
#   }
#   if(missing(statType)){stop("Type of test statistic is needed as statType. One of I, n, and G is available.")}
#   if(!statType%in%c("I", "n", "G")){stop("Input statType is wrong. One of I, n, and, G is available.")}
#   if(statType%in%c("n", "G") & missing(n)){stop("n is required.")}
  
#   # if(missing(adjType)){stop("Type of estimate adjustment is needed as adjType. One of s and a is available.")}
#   # if(!adjType%in%c("s", "a")) {stop("Type of estimate adjustment is wrong. One of s and a is available.")}
  
#   library(dplyr)
#   # I uses sample size nU (nU * pi_U)
#   # n uses sample size n (n * pi_U)
#   # G uses sample size n and pi_G which is projected to have sum 1
#   nU = sum(u)
#   matU = matrix(u, ncol=c)
  
#   if(statType%in%c("I", "n")){
#     hat.pU = u/nU
#     hat.u.pi_1 = apply(matU, 1, function(x) sum(x)/nU)
#     hat.u.pi_2 = apply(matU, 2, function(x) sum(x)/nU)
#     hat.piU = kronecker(hat.u.pi_2, hat.u.pi_1)
#     if(!statType%in%c("n")){testStatI = nU*sum((hat.pU - hat.piU)^2/hat.piU)}
#     if(!statType%in%c("I")){testStatnU = sum((u-hat.piU*n)^2/(hat.piU*n))}
#     hat.pi_1 = hat.u.pi_1; hat.pi_2 = hat.u.pi_2
#   } 
  
#   if(statType%in%c("G")){
#     hat.g.pi_1 = apply(matU, 1, function(x) sum(x)/n - (nU-n)/(n*r))
#     hat.g.pi_2 = apply(matU, 2, function(x) sum(x)/n - (nU-n)/(n*c))
#     hat.piG = kronecker(hat.g.pi_2, hat.g.pi_1)
#     testStatnG = sum((u-n*hat.piG)^2/(n*hat.piG))
#     hat.pi_1 = hat.g.pi_1; hat.pi_2 = hat.g.pi_2
#   }
  
#   testStat = switch(statType, 
#                    "I" = testStatI,
#                    "n" = testStatnU, 
#                    "G" = testStatnG)
  
#   # hat.pi = switch(adjType, 
#   #                 "a" = list(hat.pi_1 = ifelse(hat.pi_1 < 0, abs(hat.pi_1), hat.pi_1),
#   #                            hat.pi_2 = ifelse(hat.pi_2 < 0, abs(hat.pi_2), hat.pi_2) ),
#   #                 "s" = list(hat.pi_1, hat.pi_2) %>% lapply(subSmall) # more complicated need to adjust it later.
#   #                 )
  
#   hat.pi = list(hat.pi_1, hat.pi_2)
#   return(list(testStat = testStat, hat.pi = hat.pi, statType = statType))
#   # return(list(testStat = testStat, hat.pi = hat.pi, statType = statType, adjType = adjType))
#   }

# priv.chisq.test = function(testStat, mu, sig, n, sens, adjType){
#   if( sum(!(names(testStat)%in%c("testStat", "hat.pi", "statType"))) ) {stop("Inputs are not correct: 'testStat' results are required.") }
#   if((missing(mu) && missing(sig)) ){stop("At least one mu or sig is required!")}
#   if(!missing(mu) && (missing(n)||missing(sens)) ){stop("Only mu is provided, we need 'n' and 'sens'.")}
#   if(missing(sig)){
#     sig = round(Sig(mu, n, sens), 4)
#   }
#   library(mgcv)
  
#   test.stat = testStat$testStat
#   adj.hat.pi = lapply(testStat$hat.pi, prob.adj, adjType=adjType)
#   hat.pi1 = adj.hat.pi[[1]]$hat.pi; hat.pi2 = adj.hat.pi[[2]]$hat.pi
  
#   Cov = SigInG(hat.pi1, hat.pi2, sig, type=testStat$statType)
#   lambdas = eigen(Cov, only.values = T)$values
#   pval = psum.chisq(test.stat, lb=lambdas)
#   output = list(Sigma = Cov, lambda = lambdas, testStat = test.stat, hat.pi = list(hat.pi1, hat.pi2),
#                 p.value = pval, statType = testStat$statType, adjType = adjType)
#   return(output)
# }

# # quantile computation
# quantile.chisq = function(testStat, lambda){
#   library(mgcv)
#   tmp.stat = testStat
#   p = psum.chisq(tmp.stat, lb=lambda)
#   while(p < 0.99){
#     tmp.stat = tmp.stat/(1-p)
#     p = psum.chisq(tmp.stat, lb=lambda)
#   }
#   seq.stat = seq(0, tmp.stat, length=300)
#   q = psum.chisq(seq.stat)
#   q1 = max(q[q <= .25])
#   q2 = max(q[q <= .5])
#   q3 = max(q[q <= .75])
#   q4 = max(q)
#   return(list(q1=q1,q2=q2,q3=q3,q4=q4))
# }

# # plot drawing 
# pdf.null = function(testStat, lambda){
#   library(mgcv)
#   q = quantile.chisq(testStat, lambda)
#   q4 = q$q4
#   pnts = seq(0, q4, length = 10000)
#   h = unique(round(diff(pnts), 5))
  
#   if(length(h)!= 1) h=min[h]
  
#   cdf = psum.chisq(pnts, lb = lambda)
#   pdf = diff(cdf)/h
  
#   pdfpnts = cbind(pnts, c(0, pdf) )
#   plot(pdfpnts, type="l")
#   return(pdfpnts)
# }

# # prob adjust
# prob.adj = function(hat.pi, adjType){
#   if(missing(adjType)){stop("Type of estimate adjustment is needed as adjType. One of s and a is available.")}
#   if(!adjType%in%c("s", "a")) {stop("Type of estimate adjustment is wrong. One of s and a is available.")}
#   library(dplyr)
  
#   adj.hat.pi = switch(adjType,
#                   "a" = ifelse(hat.pi < 0, abs(hat.pi), hat.pi) %>% {./sum(.)},
#                   "s" = subSmall(hat.pi)
#                   )
#   return(list(hat.pi = adj.hat.pi, adjType=adjType))
# }