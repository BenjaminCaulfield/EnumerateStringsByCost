import math
import heapq
import copy
import cfg
import time
import sympy

#cost value that caps the number of occurrences of different symbols
#also maintains a "main cost", so the cost used by the algorithm is the main cost unless the symbol cap is reached.
#
class symCap:
    cost = 0
    caps = {} #caps[f] is the number of occurences of f that are allowed.
    occurs = {} #counts the number of occurrences for each symbol

    
    def __init__(self, symCaps, initCost = 0, initOccurs = False):
        self.cost = initCost
        self.caps = symCaps
        if initOccurs:
            self.occurs = initOccurs
        else:
            self.occurs = {a:0 for a in symCaps.keys()}

    def __str__(self):
        retStr = "Main Cost: " + str(self.cost) + "\n"
        retStr += "Occurences/Caps: "
        for a in self.caps:
            retStr += str(a) + ":" + str(self.occurs[a]) + "/" + str(self.caps[a]) + " "
        return retStr + "\n"

    def __hash__(self):
        return hash(self.cost) + sum([ hash(key)*hash(value) for key, value in self.occurs.iteritems()])

    def applyFunc(self, func, costs, sym):
        newCost = func([c.cost for c in costs])

        newOcc = {}
        for a in self.caps:
            newOcc[a] = sum([c.occurs[a] for c in costs])
            if sym == a:
                newOcc[a] = newOcc[a] + 1
            if newOcc[a] > self.caps[a]:
                newCost = float('inf')

        return symCap(self.caps, newCost, newOcc)

    def __lt__(self, other): #compares costs lexicographically with the main cost first
        if other == float('inf'):
            return True
        if other == float('-inf'):
            return False
        if self.cost < other.cost:
            return True
        if self.cost > other.cost:
            return False
        for s in self.caps:
            if self.occurs[s] < other.occurs[s]:
                return True
            if self.occurs[s] > other.occurs[s]:
                return False
        return False #they're equal
    def __eq__(self, other):
        if other == float('inf') or other == float('-inf'):
            return False
        assert self.caps == other.caps
        return self.cost == other.cost and self.occurs == other.occurs


class DerivationTree:
    root = "S" #Name of root nonterminal
    func = ("a", 0) #function name and arity
    cost = None #Stores cost of tree
    costFunc = lambda x: x #Cost function: x is a list of costs of subtrees
    children = [] #list of children that are derivation trees
    hashVal = None #!! be careful, if children are changed, hashVal should too.
    termHashVal = None #hash value that doesn't consider nonterminals

    assert len(children) == func[1]

    def __init__(self, rootSymbol = "S", function = ("a", 0), costFunction = lambda x: x, childs = [], getCost = False):
        self.root = rootSymbol
        self.func = function
        self.costFunc = costFunction
        self.children = childs
        #hash(self)
        if getCost:
            self.computeCost()


    #checks if two derivation trees are equal
    #doesnt check cost functions (Turing dropped the ball)
    #trees with same function struction, but different nonterminals are not equivalent.
    def __eq__(self, other, checkCost = False):
        if self is other:
            return True
        return hash(self) == hash(other) #!! should we be worried about two things hashing to same value?
##        if(checkCost):
##            if(not self.cost):
##                self.computeCost()
##            if(not other.cost):
##                other.computeCost()
##            assert self.cost == other.cost
##
##        #if( (self.root != other.root) or (self.func != other.func)):
##        #     return False
##
##        if( self.func != other.func):
##            return False
##
##        assert len(self.children) == len(other.children) #ow, func should be different
##
##        #checks if children are all equal
##        return all(map(lambda x: self.children[x] == other.children[x], range(0, len(self.children))))        

    def __str__(self):
        return self.derStr(False)

    #!! can be made much more efficient by storing hashvalues. 
    #checks if derivation trees yield the same term (even if through different nonterminals/productions)
    def sameTerm(self, other):
        if self.func != other.func:
            return False
        if self == other: #checks if they are completely equal
            return True
        return all(map(lambda x: self.children[x].sameTerm(other.children[x]), range(0, len(self.children))))        


        
##    def __hash__(self):
##        if self.hashVal:
##            return self.hashVal
##        else:
##            hashV = hash(self.root) * hash(self.func)
##            for c in self.children:
##                hashV += hash(c)
##            self.hashVal = hashV
##            return hashV

    #nonT is true if the nonterminals are included in the resulting string
    def derStr(self, nonT = True):
        string = ""
        if(nonT):
            string += self.root + ":"
        string += self.func[0]
        if len(self.children) == 0:
            return string
        string += "["
        for c in self.children:
            string += c.derStr(nonT) + ", "
        string = string[:-2] + "]"
        return string

    def __hash__(self):
        if self.hashVal:
            return self.hashVal
        h = hash(self.func) + hash(self.root) #hash(str(self))
        for i in range(0, len(self.children)):
            h += hash(self.children[i])*sympy.prime(i+10) #guarantees hash function is different if order of children is swapped
#        if self.cost:
#            h += hash(self.cost)
        self.hashVal = h
        return h

    #yields a hash value that only considers the term produced by the derivation tree (not the nonterminals/productions)
    def termHash(self):
        if self.termHashVal:
            return self.termHashVal
        h = hash(self.func) #+ hash(self.root)
        for i in range(0, len(self.children)):
            h += self.children[i].termHash()*sympy.prime(i+20) #guarantees hash function is different if order of children is swapped
#        if self.cost:
#            h += hash(self.cost)
        self.termHashVal = h
        return h

    #computes cost of tree
    #If checkCost = True, checks that computed cost of tree equals 'cost'
    def computeCost(self, checkCost = False):
        #only calculates cost if it is uncalculated or if it is checking correctness
        if((not checkCost) and self.cost):
            return self.cost
        newCost = 0
        if self.func[1] == None:
            newCost = self.costFunc([])
        else:
            args = []
            for x in self.children:
                args.append(x.computeCost(checkCost))
            newCost = self.costFunc(args)
        if(checkCost):
            assert newCost == self.cost, "new cost: " + str(newCost) + " old cost: " + str(self.cost)
        else:
            self.cost = newCost
        return self.cost

    def __lt__(self, other):
        if not self.cost:
            self.computeCost()
        if not other.computeCost():
            other.computeCost()
        if self.cost < other.cost:
            return True
        if self.cost > other.cost:
            return False
        return hash(self) < hash(other)

    #Returns root production of the form [N, f, [X1, X2, ...]] N is root, f is function symbol, [X1,...] is RHS
    def getRootProd(self):
        return [self.root, self.func[0], [C.root for C in self.children]]


#A derivation tree, but it can have nonterminals or variables at the leaves (ignore the self.root value)
#Used in let statements
class VarExpression(DerivationTree):
    variables = [] #variables appearing at leaves in expression
    nonterminals = [] #nonterminals appearing at leaves in expression
    foundVars = False

    #updates list of variables and nonterminals appearing at roots in tree
    def getVarsNonTs(self, allVars, nonTerms):
        self.foundVars = True
        self.variables = set()
        self.nonterminals = [] #counts repeats of non-terminals but not variables
        if self.children == []:
            if self.func[0] in allVars:
                self.variables.add(self.func[0])
            elif self.func[0] in nonTerms:
                self.nonterminals.append(self.func[0])
        else:
            for child in self.children:
                child.getVarsNonTs(allVars, nonTerms)
                self.variables.update(child.variables)
                self.nonterminals += child.nonterminals
        self.variables = list(self.variables)

    #the cost function for the new flattened symbol
    #x is concatenated list of nonterminals-values and variable values
    #passes values of all variables and only values of relevant non-terminals to each child
    #getVarsNonTs must have been called for it to work
    def flattenedFunc(self, x):
        assert self.foundVars, "getVarsNonTs must be called first"
        varVals = {self.variables[i]:x[i + len(self.nonterminals)] for i in range(0, len(self.variables))}
        nonTs = x[0:len(self.nonterminals)]
        if self.children == []:
            if self.func[0] in varVals:
                assert nonTs == []
                return varVals[self.func[0]]
            elif self.func[0] in self.nonterminals:
                assert len(nonTs) == 1
                return nonTs[0]
            else:
                return self.costFunc([])
        subVals = []
        for child in self.children:
            subVals.append(child.flattenedFunc(nonTs[0:len(child.nonterminals)] + [varVals[v] for v in child.variables]))
            nonTs = nonTs[len(child.nonterminals):]
        return self.costFunc(subVals)

    #the cost function for the new flattened symbol
    #x is concatenated list of nonterminals-values
    #passes  values of relevant non-terminals to each child
    #getVarsNonTs must have been called for it to work
    def oldflattenedFunc(self, x):
        assert self.foundVars, "getVarsNonTs must be called first"
        #varVals = {self.variables[i]:x[i + len(self.nonterminals)] for i in range(0, len(self.variables))}
        #nonTs = x[0:len(self.nonterminals)]
        if self.children == []:
            if self.func[0] in self.nonterminals:
                assert len(nonTs) == 1
                return nonTs[0]
            else:
                return self.costFunc([])
        subVals = []
        for child in self.children:
            subVals.append(child.flattenedFunc(nonTs[0:len(child.nonterminals)]))
            nonTs = nonTs[len(child.nonterminals):]
        return self.costFunc(subVals)

    #returns a varExpression with all leaves called varName replaced by repExpression
    #NOTE: can make more efficient by having it handle multiple replacements at once
    def replaceVar(self, varName, repExpression):
        if self.func[0] == varName:
            return copy.deepcopy(repExpression)
        else:
            newchildren = []
            for child in self.children:
                newchildren.append(child.replaceVar(varName, repExpression))
            self.children = newchildren
        return self
            
class CFG:
    testCons = True #runs test to make sure everything is consistent
    root = "S" #root nonterminal
    nonterminals = ["S"] #list of nonterminal names
    alphabet = {} #maps names of terminals to rank (arity) 
    productions = [] #list of productions of the form [N, f, [X1, X2, ...]] N is root, f is function symbol, [X1,...] is RHS
    costs = {} #maps function symbols to superior functions from list(float) -> float

    let_productions = [] #list of productions/ let-expressions of form [N, [[v1, e1], [v2, e2],...], f, [e1', e2',....]]  
    #The pairs [vi, ei] are the settings let(vi = ei) in the let statements
    #the e's are lists of form [f, [e'_1, ...]] containing variables, nonterminals, and terminals at the leaves

    def __init__(self):
        self.testCons = True
        self.root = "Start"
        self.nonterminals = ["Start"]
        self.alphabet = {}
        self.productions = []
        self.costs = {}

    #checks alphabet, costs, productions, nonterminals, and root are consistent with each other
    #returns True if they are, otherwise makes them consistent
    #doesn't check that every nonterminal/symbol appears in a production
    def makeConsistent(self):
        consistent = True
        if self.root not in self.nonterminals:
            self.nonterminals.append(root)
            consistent = False
        nonTset = set(self.nonterminals)
        oldProductions = self.productions
        self.productions = []
        for prod in oldProductions:
            if prod[0] not in nonTset:
                nonTset.add(prod[0])
                consistent = False
            if prod[1] not in self.alphabet:
                self.alphabet[prod[1]] = len(prod[2])
                consistent = False
            assert self.alphabet[prod[1]] == len(prod[2]), "multiple arities listed for " + prod[1]
        for prod in oldProductions:
            newRHS = []
            for nonT in prod[2]:
                if nonT not in nonTset: #assumes nonT is terminal. Creates new nonterminal A_nonT and production pointing to nonT
                    self.alphabet[nonT] = 0
                    newNT = "A_" + nonT
                    newRHS.append(newNT)
                    nonTset.add(newNT)
                    newProd = [newNT, nonT, []]
                    if newProd not in self.productions:
                        self.productions.append(newProd)
                    consistent = False
                else:
                    newRHS.append(nonT)
            self.productions.append([prod[0], prod[1], newRHS])
        self.nonterminals = list(nonTset)
#        caps = {a[0]:1 for a in self.alphabet}
 #       cap = symCap(caps)
        return consistent

    def setCostsToDepth(self):
        for sym in self.alphabet:
            if sym not in self.costs.keys():
                if self.alphabet[sym] == 0:
                    self.costs[sym] = lambda x: 1
                else:
                    self.costs[sym] = lambda x: 1 + max(x) #checks depth
        #for cost in self.costs:
#            assert cost in self.alphabet

    def setCostsToSize(self):
        for sym in self.alphabet:
            if sym not in self.costs.keys():
                if self.alphabet[sym] == 0:
                    self.costs[sym] = lambda x: 1
                else:
                    self.costs[sym] = lambda x:1 + sum(x) #checks size

    def setCostsWithCaps(self, cap):
        capper = symCap({a:cap for a in self.alphabet})
        for sym in self.alphabet:
            if sym not in self.costs.keys():
                if self.alphabet[sym] == 0:
                    self.costs[sym] = lambda y: capper.applyFunc(lambda x: 1, y, sym)
                else:
                    self.costs[sym] = lambda y: cap.applyFunc(lambda x: sum(x) + 1, y, sym) #checks size
                    #self.costs[sym] = lambda y: cap.applyFunc(lambda x: max(x) + 1, y, sym) #checks depth
        for cost in self.costs:
            assert cost in self.alphabet

        

    #helper function to processLetStatements
    #finds location of all variables and let statements in expression [f, [e'_1, ...]] 
    #represents locations as list of children to follow from root e.g., location of x in f(a, g(x,b)) is [1, 0]
    def findVars(self, vNT, expression):
        varIndices = {v:[] for v in vNT}
        index = []
        subExp = [expression] #pointers to all subexpressions of currently examined expression
        while True:
            print str(["XXXX" + str(x) + "BBB" for x in subExp])
            if subExp[-1][1] == []: #If at a leaf
                if subExp[-1][0] in vNT:
                    varIndices[subExp[-1][0]].append(copy.copy(index))
                subExp = subExp[:-1] #pops and points parent to next child.
                index[-1] = index[-1] + 1
            else:
                if len(subExp) == len(index) and index[-1] == len(subExp[-1][1]): #all children are seen. pops and points parent to next child
                    subExp = subExp[:-1]
                    index = index[:-1]
                    if subExp == []:
                        return varIndices
                    index[-1]+=1
                    continue
                if len(subExp) == len(index) + 1: #index doesn't point to next child
                    index.append(0)
                assert len(subExp) == len(index), [subExp] + [index]
                assert len(subExp[-1][1]) > index[-1]
                subExp.append(subExp[-1][1][index[-1]])
#WHAT TO DO HERE                subExp.append(subExp[-1][1][index[-1]])
                
        

    #adds productions that are equivalent to the statements in let-productions
    # let-productions are of form [N, [[v1, e1], [v2, e2],...], e]
    def oldProcessLets(self):
        #for ind in range(0,len(self.let_productions)):
        for letProd in self.let_productions:
            print letProd
            varDefs = letProd[1]
            variables = [p[0] for p in varDefs]
            for ind in range(0,len(varDefs)):
                varDef = varDefs[-(ind+1)] #runs through list in reverse since later expressions might include earier variables
                if letProd[2].func[0] in variables: #expression on rhs is a variable
                    assert len(letProd[2].children) == 0
                    assert varDef[0] == letProd[2].func[0] #current variable defined is rhs of let-expression
                    letProd[2].func = (varDef[1].func[0], len(varDef[1].children)) #replaces variable with variable rhs of let-expression
                    letProd[2].children = copy.copy(varDef[1].children)
                    #letProd[2].children = copy.deepcopy(varDef[1].children) #DO I NEED DEEPCOPY HERE?
                else:
                    newSubs = []
                    for subExpr in letProd[2].children:
                        newSubs.append(subExpr.replaceVar(varDef[0], varDef[1]))
                    letProd[2].children = newSubs
            #replace expression with single function call & flattened cost function
            

            newSym = letProd[2].func[0] + "_" + str(hash(letProd[2])) #Warning:hash function is not deterministic
            letProd[2].getVarsNonTs(variables, self.nonterminals)
            self.alphabet[newSym] = len(letProd[2].variables) + len(letProd[2].nonterminals)
            newProd = [letProd[0], newSym, ]
            return letProd[2]


        #adds productions that are equivalent to the statements in let-productions
    # let-productions are of form [N, [[v1, e1], [v2, e2],...[vk,ek]], e]
    def processLets(self):
        #for ind in range(0,len(self.let_productions)):
        for letProd in self.let_productions:
            print letProd
            varDefs = copy.copy(letProd[1]) #pairs [v1, e1]
            variables = [p[0] for p in varDefs]
            if letProd[2].func[0] in variables:
                assert varDefs[-1][0] == letProd[2].func[0] #e is a variable, assert it's the last variable (vk = ek)
                varDefs = varDefs[:-1] + [[None, varDefs[-1][1]]] #treats last var def ek as if it were e
                variables.remove(letProd[2].func[0]) #removes vk from variables
            else:
                varDefs.append([None, letProd[2]])
            print varDefs
            varNonTs = {v:v+"_"+str(sum([hash(varDef[1]) for varDef in varDefs])%10000) for v in variables} 
            for varDef in varDefs:

                assert not varDef[1].func[0] in variables, "found variable definition vi = vj" 

                nonT = None #nonT is nonterminal on LHS of production
                if varDef[0]:
                    nonT = varNonTs[varDef[0]]
                else:
                    nonT = letProd[0]

                #handles case when zi = A. Adds production z-> A. Later copies all A -> f(..) to zi -> f(..)
                if varDef[1].func[0] in self.nonterminals: 
####                    for prod in self.productions:
####                        if prod[0] == varDef[1].func[0]:
####                            newProd = copy.deepcopy(prod)
####                            newProd[0] = nonT
####                            self.productions.append(newProd)
####                    for prod in self.let_productions:
####                        if prod[0] == varDef[1].func[0]:
####                            newProd = copy.deepcopy(prod)
####                            newProd[0] = nonT
####                            self.let_productions.append(newProd)
                    print "R", nonT
                    if nonT == varDef[1].func[0]:
                        continue
                    self.productions.append([nonT, varDef[1].func[0], []]) #production of form z -> A
                    continue


                
                newSym = varDef[1].func[0] + "_" + str(hash(varDef[1])%10000) #Warning:hash function is not deterministic
                varDef[1].getVarsNonTs(variables, self.nonterminals)

                #for v in varDef[1].variables:
#                    newVar = VarExpression(function = (varNonTs[v], 0))
#                    varDef[1].replaceVar(v, newVar)
                
                #varDef[1].getVarsNonTs([], self.nonterminals)
                print newSym, varDef
                self.alphabet[newSym] =len(varDef[1].variables) + len(varDef[1].nonterminals)
                self.costs[newSym] = varDef[1].flattenedFunc

                newProd = [nonT, newSym, varDef[1].nonterminals + [varNonTs[v] for v in varDef[1].variables]]
                self.productions.append(newProd)
            self.nonterminals += varNonTs.values()
        self.nonterminals = list(set(self.nonterminals)) #removes duplicates
        newProductions = []
        badProductions = []
        for prod in self.productions: #handles productions A -> B
            if prod[1] in self.nonterminals:
                for prod2 in self.productions:
                    if prod2[0] == prod[1] and prod2[1] not in self.nonterminals:
                        newProductions.append([prod[0], prod2[1], prod2[2]])
                badProductions.append(prod)
        for p in badProductions:
            self.productions.remove(p)
        self.productions += newProductions

    #checks if grammar structure and func names are same
    #doesn't check if cost funcs are same
    #if nonterminal names are different, the grammars are considered different
    def __eq__(self, other):
        if self.root != other.root:
            return False
        selfN = set(self.nonterminals)
        otherN = set(other.nonterminals)
        if selfN != otherN:
            return False
        selfA = set(self.alphabet)
        otherA = set(other.alphabet)
        if selfA != otherA:
            return False
        selfP = set([(x[0], x[1], tuple(x[2])) for x in self.productions])
        otherP = set([(x[0], x[1], tuple(x[2])) for x in other.productions])
        if selfP != otherP:
            return False
        return True
        
    def __str__(self):
        returnStr = "Alphabet: " + str(self.alphabet) + "\n"
        returnStr += "Root: " + self.root + "\n"
        returnStr += "Nonterminals: " + str(self.nonterminals) + "\n"

        sortedProds = {n:[] for n in self.nonterminals}
        for p in self.productions:
            sortedProds[p[0]].append((p[1],p[2]))
        returnStr += "Productions: \n"
        for p in sortedProds:
            returnStr += "    " + p + " -> "
            for x in sortedProds[p]:
                returnStr += str(x[0]) 
                if len(x[1]) > 0:
                    returnStr += "("
                for y in x[1]:
                    returnStr += str(y) + ","
                if len(x[1]) > 0:
                    returnStr = returnStr[:-1]
                    returnStr += ")"
                returnStr += " | " 
            if returnStr[-2] == "|":
                returnStr = returnStr[:-2]
            returnStr += "\n"
        returnStr += "\n"
        return returnStr
        
    
    def testCFG(self):
        assert self.root in self.nonterminals
        constant = False
        for prod in self.productions:
            assert prod[1] in self.alphabet
            assert len(prod[2]) == self.alphabet[prod[1]]
            constant = constant or len(prod[2])==0
        assert constant
        for a in self.alphabet:
            assert a in self.costs
    def Knijkstra(self): 
        mu = {} #maps nonterminals to min weight of subtrees from them
        minprods = {} #maps nonterminals to minimal transition from them
        D = set() # list of nonterminals for which the minimal transition has been found
        Dlist = [] #D as a list ordered by when elts are added to D
        nProds = {} #maps nonterminal N to list of productions st N appears on RHS of production
        heap = [] #contains tuple of mu value and relevant production

        for n in self.nonterminals:
            mu[n] = float("inf")
            nProds[n] = []


        for p in self.productions:
            if not p[2]: #function has arity 0 so we push to heap
                #print(p)
#                heapq.heappush(heap, (float(self.costs[p[1]]([])), p))#!!removed float
                heapq.heappush(heap, (self.costs[p[1]]([]), p)) 

            for r in p[2]:
                nProds[r].append(p)
        while len(D) < len(self.nonterminals):
            (val, prod) = heapq.heappop(heap)
    

            #print "current production: ", val, prod
            #print mu
            #print minprods
            
            if prod[0] in D: #for now heap can contain multiple copies of same nonterminal
                continue
            mu[prod[0]] = val
            for uProd in nProds[prod[0]]: #checks each production with prod[0] on rhs to see if entire rhs is minimized
                rset = set(uProd[2])
                rset.remove(prod[0])
                if rset.issubset(D): #new production with all rhs nonterminals minimized
                    Dvals = [mu[N] for N in uProd[2]]
                    #prodCost = float(self.costs[uProd[1]](Dvals))!! removed float
                    prodCost = self.costs[uProd[1]](Dvals)
                    heapq.heappush(heap, (prodCost, uProd))
            minprods[prod[0]] = prod
            D.add(prod[0])
            Dlist.append(prod[0])
        return (mu, minprods, Dlist)
    #takes set of derivations trees and dict mapping costs to their frequency
    #checks number of derivation of each cost matches frequencies
    def checkCostFrequency(self, derList, freqs):
        seenFreqs = {k:0 for k in freqs}
        for der in derList:
            dCost = der.computeCost()
            assert dCost in freqs, "Derivation Tree: " + str(der) + " has invalid cost " + str(dCost)
            seenFreqs[dCost] += 1
        assert seenFreqs == freqs

    #determines the min distance from root to each other nonterminal
    #returns dict {nonterm:dist}
    def getDistFromRoot(self):
        distFromRoot = {self.root:0}
        queue = [self.root]
        seenTerms = set([self.root])
        numTerms = 0
        prods = {n:[] for n in self.nonterminals}
        for p in self.productions:
            prods[p[0]].append(p[2])
        while numTerms < len(queue):
            nt = queue[numTerms]
            for rhs in prods[nt]:
                for r in rhs:
                    if r not in seenTerms:
                        queue.append(r)
                        distFromRoot[r] = distFromRoot[nt] + 1
                        seenTerms.add(r)
            numTerms += 1
        for n in self.nonterminals:
            assert n in queue, "not all nonterminals are reachable from start"
        return distFromRoot

    #helper func for enumByDepth
    #takes in a list of numbers [x_1 ... x_k]
    #returns all lists (t_1, ..., t_k) where  0 <= t_i < x_i
    def crossProduct(self, sizes):
        prods = [[i] for i in range(0, sizes[0])]
        for i in range(1, len(sizes)):
            newProds = []
            for p in prods:
                for j in range(0, sizes[i]):
                    newP =copy.copy(p)
                    newP.append(j)
                    newProds.append(newP)
            prods = newProds
        prods = [tuple(p) for p in prods]
        return prods

            
        
            
        
    #enumerates all trees u to a fixed depth
    def enumByDepth(self, d):
        dists = self.getDistFromRoot()
        #pass to get dist from root
        prods = {n:[] for n in self.nonterminals}
        for p in self.productions:
            prods[p[0]].append((p[1],p[2]))

        
        #treeDepths[A][d] yields list of trees of depth <= d starting from A
        treeDepths = {A:[[] for _ in range(0,d-dists[A]+1)] for A in self.nonterminals}
        curr_depth = 0
        while curr_depth <= d:
            for A in self.nonterminals:
                if dists[A] + curr_depth <= d:
                    if curr_depth > 0:
                        treeDepths[A][curr_depth] = copy.copy(treeDepths[A][curr_depth-1]) #shouldn't need copy here
                    for p in prods[A]:
                        dt = DerivationTree(A, (p[0], len(p[1])), (lambda x: 1 + max(x + [-1]))) #uses depth as cost 
                        if curr_depth == 0:
                            if len(p[1]) == 0:
                                treeDepths[A][curr_depth].append(dt)
                            continue
                        if len(p[1]) == 0:
                            continue
                        numSubtrees = [len(treeDepths[B][curr_depth-1]) for B in p[1]]
                        if 0 in numSubtrees:
                            continue
                        print numSubtrees
                        
                        indices = self.crossProduct(numSubtrees)
                        for inds in indices:
                            #print "ind: " + str(inds)
                            newChildren = [treeDepths[p[1][i]][curr_depth-1][inds[i]] for i in range(0, len(p[1]))]

                            #checks that the constructed tree is of depth curr_depth (and thus is new)
                            fullDepth = False
                            for child in newChildren:
                                assert child.cost != None
                                assert child.cost <= curr_depth-1, str(child.cost) + str(child)
                                if child.cost == curr_depth - 1:
                                    fullDepth = True
#                                    print str(child) + " " + str(child.cost)
                            if not fullDepth:
#                                print "not deep" + str(curr_depth)
 #                               for x in newChildren:
#                                    print str(x) + " " + str(x.cost)
                                continue

 #                           print "SDFSDFSDF" + str(fullDepth)
                            newdt = copy.copy(dt)
                            newdt.children = newChildren
 #                           print "new tree:" + str(newdt) + " " + str(curr_depth)
                            newdt.computeCost()
                            treeDepths[A][curr_depth].append(newdt)
                            #if not (newdt in treeDepths[A][curr_depth]): #inefficient to check every time
                                #treeDepths[A][curr_depth].append(newdt)
            curr_depth += 1
        return treeDepths

    #recursive helper for enumerating from a given non-terminal A
    #def EnumByDepthRec(self, A, d)
#        trees = {}
        
        
    def getTreesFromProds(self, prods, order):
        trees = {}
        for Y in order:
            trees[Y] = DerivationTree(prods[Y][0],
                                     (prods[Y][1], len(prods[Y][2])),
                                     self.costs[prods[Y][1]],
                                     [trees[Z] for Z in prods[Y][2]])
##                                     
##                                     
##            newTree = DerivationTree()
##            newTree.root = prods[Y][0]
##            newTree.func = (prods[Y][1], len(prods[Y][2]))
##            newTree.costFunc = self.costs[newTree.func[0]]
##            newTree.children = [trees[Z] for Z in prods[Y][2]]
##            newTree.computeCost()
##           trees[Y] = newTree
        return trees


    def EnumerateStrings(self, k):
        (mu, minprods, order) = self.Knijkstra()
        mintrees = self.getTreesFromProds(minprods, order)

        timesPushed = 0

        #for dt in mintrees.values():
#            print dt.root, dt, hash(dt), dt.termHash()

        #Set of trees that are min or have one non-optimal production (At the root)
        H_ = {Y:DerStruct(k, []) for Y in order}
        #H_ = {Y:DerStruct(k, [mintrees[Y]]) for Y in order} #!!can remove sets from derstruct

        F = TermStorage(mintrees)
        for p in self.productions:
            T = DerivationTree(p[0],
                      (p[1], len(p[2])),
                      self.costs[p[1]],
                      [copy.deepcopy(mintrees[c]) for c in p[2]],
                      True)#!! do we need a deepcopy?
            #print p[0], T
            if not F.contains(T): #!! add to alg
                H_[p[0]].dPush(T)
                #print "init: ", T.root, T, hash(T), T.termHash()

        #changedNonTsLast = {N:True for N in self.nonterminals} #tracks which nonterminals had updated heaps in the last round
        #changedNonTsCurrent = {N:False for N in self.nonterminals} #same, but in current round

        doneNT = {N:False for N in self.nonterminals} #tracks which nonterminals have found all possible trees
        numComplete = 0 
        
        for j in range(2,k+1):
            #assert F.numTrees("Start") == j-1, str(j) + str(F)
            if numComplete == len(self.nonterminals): #stops if all nonterminals have found minimal trees (used in dags)
                assert False
                break
            #seenNewDer = False
            for Y in order:
                if doneNT[Y]:
                    continue

                T_ = F.getLast(Y)
                for l in range(0, len(T_.children)):
                    #assert F_dict[T_.children[l].root].get(T_.children[l]), str(T_) + " " + str(T_.children[l]) + " " + str([l_.root for l_ in T_.children])
                    tl_index = F.getPos(T_.children[l])# F_dict[T_.children[l].root][T_.children[l]]

                    Z_l = T_.children[l].root

                    #print T_

                    if F.numTrees(Z_l) > tl_index + 1: #len(F_order[T_.children[l].root]) > Tl_index + 1:
                        newChildren = copy.copy(T_.children)
                        newChildren[l] = F.get(Z_l, tl_index+1) #F_order[T_.children[l].root][Tl_index+1]  #!! should we copy this tree with : copy.copy(Tl)                           
                        T_2 = DerivationTree(T_.root,
                                            T_.func,
                                            T_.costFunc,
                                            newChildren,
                                            True)
                        H_[Y].dPush(T_2)
#                        print "pushing: ", T_2

                #print "T:", T_.root, T_
                #print "T_prime", T_prime
                contHere = False
                T_prime = False
                while True:
                    if H_[Y].empty(): #!! add to alg?
                        doneNT[Y] = True
                        numComplete += 1
                        contHere = True
                        break
                    T_prime = H_[Y].dPop()
                    if not F.contains(T_prime): break
                if contHere: continue
                assert T_prime
                F.push(T_prime)
        #print F
        return F.F_order[self.root]                   


#stores and looks up trees.
#tracks which trees appear from which nonterminals, but considers trees with same terms and different nonterminals to be the same
class TermStorage:
    F_order = {}  #F_order[Y] is ordered list of mincost trees from Y
    F_dict = {}   #F_dict[Y][T.termHash()] is the position of T in F_order[Y] 

    def __init__(self, mintrees):
        for A in mintrees:
            self.F_order[A] = []
            self.F_dict[A] = {}
            self.push(mintrees[A])
            #self.F_order[A] = [mintrees[A]]
            #self.F_dict[A] = {mintrees[A].termHash():0} 

    def push(self, T):
        assert T != None
        if not T:
            for v in F_order:
                for t in v:
                    print t
            assert False
        if not T.cost:
            T.computeCost()
        self.F_order[T.root].append(T)
        self.F_dict[T.root][T.termHash()] = len(self.F_order[T.root])-1

    #returns the position of T in F_order
    def getPos(self, T):
        return self.F_dict[T.root][T.termHash()]

    #returns the ith tree in
    def get(self, A, i):
        return self.F_order[A][i]
    
    def contains(self, T):
        if T.termHash() in self.F_dict[T.root]: #!! this didn't work but should have been faster -> self.F_dict[T.root].get(T.termHash()):
            return True
        return False

    #returns highest cost tree produced from A so far
    def getLast(self, A):
        return self.F_order[A][-1]

    def numTrees(self, A):
        return len(self.F_order[A])

    def __str__(self):
        returnStr = ""
        for Y in self.F_order:
            returnStr += Y + ": \n"
            for t in self.F_order[Y]:
                returnStr += "     " + str(t) + "\n"
        return returnStr
                        
#phone-2.sl large heap size
#array_sum_2_5-Q.sl

#LIKELY NOT MOST EFFICIENT. TEST ON DIFFERENCE DATASTRUCTS. MAYBE JUST USE SETS???
#Structure used for F and F'
#maintains heap and set for efficient min and membership checks
#heap may contain duplicates, but set will not, there can be `false negatives' on inStruct
class DerStruct:
    k = -1 #tracks the value of k. tries to keep size of heap under k elements.
    maxCostDer = float('-inf') #stores the cost of the largest element (used in dPush)
    dheap = [] #used to find min-cost derivation
    dset = set() #used to check if tree is in set

    #implements a doubly linked list, pointing elts to their successor (in terms of cost)
    #frontPointers = {} #if frontPointer[x] = "first" then x is first value
    #backPointers = {}  #if frontPointer[x] = "last" then x is last value
    def __init__(self, newk, ders = []):
        self.dheap = []
        self.dset = set()
        self.k = newk
        for d in ders:
            self.dPush(d) #maxed = self.dPush(d) # works if list is sorted, o/w can't guarantee there aren't smaller elements later
            #if maxed: #stops pushing if k elements are pushed.
               # return 
    #pushes derivation to struct
    #returns False if the heap size is k and the pushed der has bigger cost than the max element in the heap (doesn't push in this case)
    #if the heap is bigger than k but the pushed element is smaller than the max size, then pushes the element (and doesnt remove larger element) #!! is it faster if we remove the larger element?
    def dPush(self, d): 
        if not d.cost:
            d.computeCost()
        if len(self.dset) >= self.k and d.cost > self.maxCostDer: #pushing a bigger element than every element in struct. and struct is maxed out.
            return False
        if(not d in self.dset):
            heapq.heappush(self.dheap, (d.cost, d))
            self.dset.add(d)
            self.maxCostDer = max(self.maxCostDer, d.cost)            
        return True
#    def inStruct(self, d):
#        return d in self.dset
    #pops min elt and removes it from set
    def dPop(self):
        d = heapq.heappop(self.dheap)
        self.dset.remove(d[1])
        if self.empty():
            self.maxCostDer = float('-inf') #resets maximum derivation if heap is empty ##!! do i need this?
        return d[1]
    def dPeek(self):
        assert len(self.dheap) > 0, "can't peek at empty list"
        return self.dheap[0][1] #just pushes tree, not tuple
    def empty(self): #dheap can contain duplicates but dset will not. so returns size of dset
        if(len(self.dheap) == 0):
            assert len(self.dset) == 0, "heap in Derstruct is empty, but set is not"
        return len(self.dset) == 0
    def printStruct(self):
        for d in self.dset:
            print(str(d.cost) + ": " + str(d))
    def getDers(self):
        return self.dset
    def copy(self): #returns a shallow copy of self
        #newStruct = DerStruct([x[1] for x in self.dheap])
        newStruct = DerStruct(self.k, [])
        newStruct.dheap = copy.copy(self.dheap)
        newStruct.dset = copy.copy(self.dset)
        newStruct.maxCostDer = self.maxCostDer
        return newStruct
"""                   
                    
                
                
#Struct with just heaps
#Structure used for F and F'
#maintains heap and set for efficient min and membership checks
#heap may contain duplicates, but set will not, there can be `false negatives' on inStruct
class DerStruct:
    dheap = [] #used to find min-cost derivation
    #dset = set() #used to check if tree is in set
    def __init__(self, ders = []):
        self.dheap = []
#        self.dset = set()
        for d in ders:
            self.dPush(d)
    #pushed derivation to struct
    def dPush(self, d):
        if not d.cost:
                d.computeCost()
        if(not (d.cost, d) in self.dheap):
            heapq.heappush(self.dheap, (d.cost, d))
#            self.dset.add(d)
    def inStruct(self, d):
        return d in dheap
        #return d in self.dset
    #pops min elt and removes it from set
    def dPop(self):
        d = heapq.heappop(self.dheap)
        #self.dset.remove(d[1])
        return d[1]
    def dPeek(self):
        assert len(self.dheap) > 0, "can't peek at empty list"
        return self.dheap[0][1] #just pushes tree, not tuple
    def empty(self): #dheap can contain duplicates but dset will not. so returns size of dset
        #if(len(self.dheap) == 0):
         #   assert len(self.dset) == 0, "heap in Derstruct is empty, but set is not"
        return len(self.dheap) == 0
    def printStruct(self):
        for d in self.dheap:
            print(str(d.cost) + ": " + str(d))
    def getDers(self):
        return self.dheap


"""

cfg1 = CFG()

expression = ['f', [['g', [['a',[]], ['x',[]]]], ['x', []], ['A', []]]]
#variables = ['x', 'A']
#cfg1.findVars(variables, expression)

expression = ['f', [['a',[]], ['x', []], ['A', []]]]
variables = ['x', 'A']
#cfg1.findVars(variables, expression)


testLets = CFG()
testLets.nonterminals = ["S"]
testLets.alphabet = {'f':2, 'a':0, 'g':1}
testLets.productions =  [["S", "a", []], ["S", "g", ["S"]]]
testLets.cost = {"f":(lambda x: x[0] + x[1] + 1), "a":(lambda x: 10), "g":(lambda x: x[0] + 1)}
testLets.let_productions = [ ["S", [ ["z", ["S",[]]]], "f", [["z",[]],["z",[]]]] ]
#testLets.processLets()

varTest1 = VarExpression(function = ("z",0), costFunction = lambda x: 0)
varTest2 = VarExpression(function = ("z2",0), costFunction = lambda x: 0)
varTest3 = VarExpression(function = ("A",0), costFunction = lambda x: 0)
varTest4 = VarExpression(function = ("f",3), costFunction = lambda x: x[0] + x[1] + x[2], childs= [varTest1, varTest2, varTest3])
varTest4.getVarsNonTs(["z","z2", "z3"], ["S", "A"])

varTest4_2 = VarExpression(function = ("f",3), costFunction = lambda x: x[0] + 2*x[1] + 3*x[2], childs= [varTest3, varTest3, varTest3])

#S -> let [z = f(A, A, A)]] z 
####letExp = [["S", [["z", varTest4_2]], copy.deepcopy(varTest1)]]
####letTest = CFG()
####letTest.nonterminals.append("A")
####letTest.let_productions = letExp
####print letTest.let_productions
####letTest.processLets()
#print letTest
#assert processedLets.func[0] == "f" and processedLets.children == varTest4.children

varTest5 = VarExpression(function = ("g",2), costFunction = lambda x: max(x)+1, childs = [varTest1, copy.deepcopy(varTest4)])
varTestZ3 = VarExpression(function = ("z3",0), costFunction = lambda x: 0)
letExp2 = [["S", [["z", varTest3], ["z2", varTest3]], varTest5],
        ["A", [["z", varTest3]], varTest1]]
letTest2 = CFG()
letTest2.nonterminals.append("A")
letTest2.productions.append(["A", "a", []])
letTest2.productions.append(["A", "g", ["A", "A"]])
letTest2.costs["a"] = lambda x: 10
letTest2.costs["f"] = lambda x: x[0] + x[1] + x[2]
letTest2.alphabet["a"] = 0
letTest2.alphabet["f"] = 3
letTest2.alphabet["SDF"] = 6
letTest2.costs["g"] = lambda x: max(x)+1
letTest2.let_productions = letExp2
#letTest2.processLets()
#print letTest2
#print letTest2.costs
#x = letTest2.EnumerateStrings(10)

###assert processedLets2.children[0].func[0]== "z"
##
##varTest6 =  VarExpression(function = ("g",2), costFunction = lambda x: sum(x))
##varTest6.children = [varTest1, varTest2]
##letExp3 = [["S", [["z2", varTest3], ["z", varTest4]], varTest6]]
##letTest3 = CFG()
##letTest3.let_productions = letExp3
##letTest3.processLets()
###assert processedLets3.children[0].children[1].func[0] == "A"




#changes to alg:
#H initialized to Min tree plus all trees with production diversions at the root
#need to track used elts of H since those can't be popped back into H
#need to find min elt in heap but also look up in set.


#nick
    #changes np-complete argument
    

#Space complexity?
#discussion k^2 is actually k*MAXTREESIZE
#Look up other papers on enumerating from cost
#cut down on space complexity
#compare our costs to Loris' QSygus paper
#handwave maxtree*k
#add section on results?
#add polydegree discussion?

