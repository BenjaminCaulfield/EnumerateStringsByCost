import math
import heapq
import copy

class DerivationTree:
    root = "S" #Name of root nonterminal
    func = ("a", 0) #function name and arity
    cost = None #Stores cost of tree
    costFunc = lambda x: x #Cost function: x is a list of costs of subtrees
    children = [] #list of children that are derivation trees

    assert len(children) == func[1]

    def __init__(self, rootSymbol = "S", function = ("a", 0), costFunction = lambda x: x, childs = []):
        self.root = rootSymbol
        self.func = function
        self.costFunc = costFunction
        self.children = childs
        self.computeCost()

    #checks if two derivation trees are equal
    #doesnt check cost functions (Turing dropped the ball)
    def __eq__(self, other, checkCost = False):
        if(checkCost):
            if(not self.cost):
                self.computeCost()
            if(not other.cost):
                other.computeCost()
            assert self.cost == other.cost

        #if( (self.root != other.root) or (self.func != other.func)):
        #     return False

        if( self.func != other.func):
            return False

        assert len(self.children) == len(other.children) #ow, func should be different

        #checks if children are all equal
        return all(map(lambda x: self.children[x] == other.children[x], range(0, len(self.children))))        

    def __str__(self):
        return self.derStr(False)

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
        return hash(str(self)) + hash(self.cost)

    #computes cost of tree
    #If checkCost = True, checks that computed cost of tree equals 'cost'
    def computeCost(self, checkCost = False):
        #only calculates cost if it is uncalculated or if it is checking correctness
        if((not checkCost) and self.cost):
            return self.cost
        newCost = 0
        if self.func[1] == None:
            print "HHEEEEEEEEEEEEEEEEEEEEE"
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

    #Returns root production of the form [N, f, [X1, X2, ...]] N is root, f is function symbol, [X1,...] is RHS
    def getRootProd(self):
        return [self.root, self.func[0], [C.root for C in self.children]]

class CFG:
    testCons = True #runs test to make sure everything is consistent
    root = "S" #root nonterminal
    nonterminals = ["S"] #list of nonterminal names
    alphabet = {} #maps names of terminals to rank (arity) 
    productions = [] #list of productions of the form [N, f, [X1, X2, ...]] N is root, f is function symbol, [X1,...] is RHS
    costs = {} #maps function symbols to superior functions from list(float) -> float
    
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
                print p
                heapq.heappush(heap, (float(self.costs[p[1]]([])), p)) 
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
                    prodCost = float(self.costs[uProd[1]](Dvals))
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

        #Set of trees that are min or have one non-optimal production (At the root)
        F1 = {Y:DerStruct([mintrees[Y]]) for Y in order}
        for p in self.productions:
            F1[p[0]].dPush(DerivationTree(p[0],
                                          (p[1], len(p[2])),
                                          self.costs[p[1]],
                                          [copy.deepcopy(mintrees[c]) for c in p[2]]))
        #old F
        #F = {Y:[DerStruct([mintrees[Y]])] for Y in order}

        F = {Y:[F1[Y]] for Y in order}
            
        

        #F1[self.root].printStruct()

        #Fprime = {Y:{0:DerStruct([mintrees[Y]])} for Y in order}
        #Fprime = {Y:[DerStruct([mintrees[Y]])] for Y in order}
        #kk = Fprime[order[0]][0].dheap

        maxIters = 1
        for j in range(1,k+1):
            print str(j) + "------------------------------------"
            maxIters = j

            #old H
            #H = {Z:copy.deepcopy(F[Z][j-1]) for Z in order}

            H = {Z:DerStruct(F[Z][j-1].dset) for Z in order}

            
            usedH = set()
            
            #H = {Z:DerStruct([mintrees[Z]]) for Z in order}
            #usedH = set()
            seenNewDer = False
            for Y in order:

                F[Y].append(DerStruct(F[Y][j-1].dset))

                #F[Y].append(copy.deepcopy(F[Y][j-1]))


                #Fprime[Y].append(copy.deepcopy(F[Y][j-1])) #possibly dont need deepcopy here. set might be wrong thing to get
                i = 0
                while i < k:
                    if(H[Y].empty()): #don't check every time?
                        break
                    T = H[Y].dPop()
                    usedH.add(T)
                    #print "T: " + str(T)
                    assert T.root == Y
                    #if(not F[Y][j].inStruct(T)):
                    #    seenNewDer = True
                    F[Y][j].dPush(T)
                    #algorithm the Tl values are taken from different sets depending on whether the root production is optimal
                    optProd = (T.getRootProd() == minprods[Y])
                    usedT = False
                    for l in range(0, T.func[1]): #for each argument to root function
                        seenDers = []
                        newDer = False
                        lDers = []
                        #print str(T)
                        if optProd:
                            lDers = F[T.children[l].root][j]
                        else:
                            #print T.getRootProd()
                            lDers = F[T.children[l].root][j-1]
                        #print str(T)
                        while not lDers.empty():
                            Tl = lDers.dPop()
                            seenDers.append(Tl)
                            #T2 = copy.deepcopy(T)
                            newChildren = copy.copy(T.children)
                            newChildren[l] = copy.copy(Tl)
                            
#                            newChildren = copy.deepcopy(T.children)
 #                           newChildren[l] = copy.deepcopy(Tl)

                            T2 = DerivationTree(T.root,
                                                T.func,
                                                T.costFunc,
                                                newChildren)
                            if(T2 == T):
                                continue
                            #print "T2: " + str(T2)
                            if((not H[Y].inStruct(T2)) and (not T2 in usedH)):
                                T2.computeCost(True)
                                #print str(T) + "--" +  str(Tl)
                                #print str(T2) + " " + str(T2.cost)
                                H[Y].dPush(T2)
                                #uncomment this#
                                #print "new"
                                #print "F[Y][j]:"
                                #F[Y][j].printStruct()
                                #print "H[Y]:"
                                #H[Y].printStruct()
                                newDer = True
                                break
                        for sD in seenDers:
                            lDers.dPush(sD)
                        if newDer:
                            usedT = True
                    if usedT: #should we just increment i regardless?
                        i += 1
                if F[Y][j].dset != F[Y][j-1].dset:
                    seenNewDer = True
            if not seenNewDer: #no new derivation was added
                break
        returnTrees = {Y:set([tree for (_,tree) in heapq.nsmallest(k , F[Y][maxIters].dheap)]) for Y in order}
        print "return vals:"
        for Y in order:                            
            print Y + ":"
            for x in returnTrees[Y]:
                print str(x.cost) + ": " + str(x) 
        return returnTrees
                        
                            

                    
                    
                
                
#LIKELY NOT MOST EFFICIENT. TEST ON DIFFERENCE DATASTRUCTS. MAYBE JUST USE SETS???
#Structure used for F and F'
#maintains heap and set for efficient min and membership checks
#heap may contain duplicates, but set will not, there can be `false negatives' on inStruct
class DerStruct:
    dheap = [] #used to find min-cost derivation
    dset = set() #used to check if tree is in set
    def __init__(self, ders = []):
        self.dheap = []
        self.dset = set()
        for d in ders:
            self.dPush(d)
    #pushed derivation to struct
    def dPush(self, d):
        if not d.cost:
                d.computeCost()
        if(not d in self.dset):
            heapq.heappush(self.dheap, (d.cost, d))
            self.dset.add(d)
    def inStruct(self, d):
        return d in self.dset
    #pops min elt and removes it from set
    def dPop(self):
        d = heapq.heappop(self.dheap)
        self.dset.remove(d[1])
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
            print str(d.cost) + ": " + str(d)
    def getDers(self):
        return self.dset



    
##a = DerivationTree()
##a.costFunc = lambda x: 10
##b = DerivationTree()
##b.costFunc = lambda x: 20
##b.func = ('b', 0)
##c = DerivationTree()
##c.children = [a, b]
##c.func = ('f', 2)
##c.costFunc = (lambda y: y[0] * y[1])
##assert c.computeCost() == 200
##
##d = DerStruct([c,b,a,c])
##assert d.dPeek() == a
##assert d.inStruct(a)
##assert d.dPop() == a
##assert d.dPop() == b
##d.dPush(a)
##assert d.dPop() == a
##assert d.inStruct(c)
##assert not d.inStruct(a)
##assert not d.empty()
##d.dPop()
##assert d.empty()
##
##a2 = copy.deepcopy(a)
##assert a == a2
##assert b != a
##assert c == c
##c2 = copy.deepcopy(c)
##assert c == c2
##assert c != b
##
##g = CFG()
##g.nonterminals = ["S", "A"]
##g.alphabet = {"f":1, "a":0}
##g.productions = [["S", "f", ["A"]], ["A", "a", []]]
##g.costs = {"f":(lambda x: x[0] + 1), "a":(lambda x: 10)}
##(mu, minprods, order) = g.Knijkstra()
##assert mu["S"] == 11
##
##trees = g.getTreesFromProds(minprods, order)
###print trees
###enums = g.EnumerateStrings(3)
##
####
###S should ignore A
##g2 = CFG()
##g2.nonterminals = ["S", "A"]
##g2.alphabet = {"f":1, "a":0}
##g2.productions = [["S", "f", ["A"]], ["A", "a", []], ["S", "a", []]]
##g2.costs = {"f":(lambda x: x[0] + 1), "a":(lambda x: 10)}
##assert g2.Knijkstra()[0]["S"]==10 #should equal 10
##
##g3 = CFG()
##g3.nonterminals = ["S", "A", "B"]
##g3.alphabet = {"f":1, "a":0, "b":0}
##g3.productions = [["S", "f", ["A"]], ["A", "a", []], ["A", "f", ["B"]], ["B", "b", []]]
##g3.costs = {"f":(lambda x: x[0] + 1), "a":(lambda x: 10), "b":(lambda x: 1)}
##assert g3.Knijkstra()[0]["S"]==3 #should be 3
##

##
##g5 = CFG()
##g5.nonterminals = ["S", "A"]
##g5.alphabet = {"f":1, "a":0}
##g5.productions = [["S", "f", ["S"]], ["S", "f", ["A"]], ["A", "a", []]]
##g5.costs = {"f":(lambda x: x[0] + 1), "a":(lambda x: 10)}
##(mu5, minprods5, order5) = g5.Knijkstra()
###enums = g5.EnumerateStrings(5)
##print "----------------"
###enums["S"].printStruct()
##
##
##g6 = CFG()
##g6.nonterminals = ["S"]
##g6.alphabet = {"f":2, "a":0}
##g6.productions = [["S", "f", ["S", "S"]], ["S", "a", []]]
##g6.costs = {"f":(lambda x: x[0] + x[1]), "a":(lambda x: 1)}
###enums = g6.EnumerateStrings(5)

##bvg = CFG()
##bvg.root = "S"
##bvg.nonterminals = ["S"]
##bvg.productions = [ ["S", "x", []],
##                    ["S", "#x0", []],#should actually be 16 0s
##                    ["S", "#x1", []],
##                    ["S", "bvnot", ["S"]],
##                    ["S", "shl1", ["S"]],
##                    ["S", "shr1", ["S"]],
##                    ["S", "shr4", ["S"]],
##                    ["S", "shr16", ["S"]],
##                    ["S", "bvand", ["S", "S"]],
##                    ["S", "bvor", ["S", "S"]],
##                    ["S", "bvxor", ["S", "S"]],
##                    ["S", "bvadd", ["S", "S"]],
##                    ["S", "if0", ["S", "S", "S"]]]
##bvg.alphabet = {p[1]:len(p[2]) for p in bvg.productions}
##bvg.costs = {a:(lambda x: 1 + sum(x)) for a in bvg.alphabet}
##bvg.EnumerateStrings(2)

##### Test Grammar 1 ##########
bvg = CFG()
bvg.root = "S"
bvg.nonterminals = ["S"]
bvg.productions = [ ["S", "x", []],
                    ["S", "#x0", []],#should actually be 16 0s
                    ["S", "shl1", ["S"]],
                    ["S", "shr1", ["S"]],
                    ["S", "bvand", ["S", "S"]],
                    ["S", "if0", ["S", "S", "S"]]]
bvg.alphabet = {p[1]:len(p[2]) for p in bvg.productions}
bvg.costs = {a:(lambda x: 1 + sum(x)) for a in bvg.alphabet}

##test1 = bvg.EnumerateStrings(6)
##bvg.checkCostFrequency(test1['S'], {1:2, 2:4})
##
##test2 = bvg.EnumerateStrings(19)
##bvg.checkCostFrequency(test2['S'], {1:2, 2:4, 3:12, 4:1})
##
##test3 = bvg.EnumerateStrings(66)
##bvg.checkCostFrequency(test3['S'], {1:2, 2:4, 3:12, 4:48})


########### Test Grammar 2 ##############
###same tree has multiple derivations 
##tg2 = CFG()
##tg2.root = "S"
##tg2.nonterminals = ["S", "A"]
##tg2.productions = [ ["A", "a", []],
##                    ["S", "f", ["A"]],#should actually be 16 0s
##                    ["A", "f", ["A"]],
##                    ["S", "f", ["S"]] ]
##tg2.alphabet = {p[1]:len(p[2]) for p in tg2.productions}
##tg2.costs = {a:(lambda x: 1 + sum(x)) for a in tg2.alphabet}
##test4 = tg2.EnumerateStrings(10)
##tg2.checkCostFrequency(test4['S'], {i:1 for i in range(2, 12)}) 
##
###example from Knuth's paper
####g4 = CFG()
####g4.root = "C"
####g4.nonterminals = ["A", "B", "C"]
####g4.alphabet = {"a":0, "b":2, "c":1, "d":3, "e":0, "f":2}
####g4.productions = [["A", "a", []], ["A", "b", ["B", "C"]], ["B", "c", ["A"]], ["B", "d", ["A", "C", "A"]], ["C", "e", []], ["C", "f", ["B", "A"]]]
####g4.costs = {"a":(lambda x: 4), "b":(lambda x: max(x[0], x[1])), "c":(lambda x: x[0] + 1), "d":(lambda x: x[0] + max(x[1],x[2])), "e":(lambda x: 9), "f":(lambda x: 0.5*(x[0] + x[1] + max(x[0], x[1])))}
####(mu4, prods4, order4) = g4.Knijkstra()
####trees4 = g4.getTreesFromProds(prods4, order4)
#####enums = g4.EnumerateStrings(3)
##
##
############################ SYGUS TESTS #########################
##
############ Test Grammar 3 #############
###multiple nonterminals, uses max cost function
##tg5 = CFG()
##tg5.root = "S"
##tg5.nonterminals = ["S", "A", "B"]
##tg5.productions = [ ["A", "a", []],
##                    ["S", "f", ["A", "B"]],
##                    ["S", "f", ["B", "A"]],
##                    #["A", "f", ["B", "A"]],
##                    #["S", "g", ["S"]],
##                    #["S", "f", ["S", "S"]],
##                    ["A", "g", ["B"]],
##                    ["B", "g", ["A"]]]
##tg5.alphabet = {p[1]:len(p[2]) for p in tg2.productions}
##tg5.costs = {'f':(lambda x: max(x)), 'g':(lambda x: x[0] + 1), 'a':(lambda x:1)}
###test5 = tg5.EnumerateStrings(5)
##tg5.checkCostFrequency(test5['S'], {2:2, 3:2, 4:1}) 
##

#### Test Grammar 4: Nand ######
##My grammar doesn't allow for productions without functions,
#Replaced vars and constants non-terminals with a production for each var & const
nand = CFG()
nand.root = "Start"
nand.nonterminals = ["Start", "StartAnd"]
nand.productions = [["Start", "a", []],
                    ["Start", "b", []],
                    ["Start", "c", []],
                    ["Start", "d", []],
                    ["Start", "true", []],
                    ["Start", "false", []],
                    ["Start", "not", ["StartAnd"]],
                    ["StartAnd", "and", ["Start", "Start"]]]
nand.alphabet = {p[1]:len(p[2]) for p in nand.productions}
nand.costs = {a:(lambda x: 1 + sum(x)) for a in nand.alphabet}
#nand.EnumerateStrings(10)

#####Test Grammar 5: ITE ############
iteg = CFG()
iteg.root = "Start"
iteg.nonterminals = ["Start", "BoolExpr"]
iteg.productions = [["Start", "0", []],
                    ["Start", "1", []],
                    ["Start", "2", []],
                    ["Start", "3", []],
                    ["Start", "4", []],
                    ["Start", "5", []],
                    ["Start", "6", []],
                    ["Start", "7", []],
                    ["Start", "8", []],
                    ["Start", "9", []],
                    ["Start", "y1", []],
                    ["Start", "y2", []],
                    ["Start", "y3", []],
                    ["Start", "y4", []],
                    ["Start", "y5", []],
                    ["Start", "y6", []],
                    ["Start", "y7", []],
                    ["Start", "y8", []],
                    ["Start", "y9", []],
                    ["Start", "z", []],
                    ["Start", "+", ["Start", "Start"]],
                    ["Start", "ite", ["BoolExpr", "Start", "Start"]],
                    ["BoolExpr", "<", ["Start", "Start"]],
                    ["BoolExpr", "<=", ["Start", "Start"]],
                    ["BoolExpr", ">", ["Start", "Start"]],
                    ["BoolExpr", ">=", ["Start", "Start"]]]
iteg.alphabet = {p[1]:len(p[2]) for p in iteg.productions}
iteg.costs = {a:(lambda x: 1 + sum(x)) for a in iteg.alphabet}
#iteg.EnumerateStrings(30)


#####Test Grammar 6: Commutative #######
comm = CFG()
comm.root = "Start"
comm.nonterminals = ["Start"]
comm.productions = [["Start", "+", ["Start", "Start"]],
                    ["Start", "-", ["Start", "Start"]],
                    ["Start", "x", []],
                    ["Start", "y", []]]
comm.alphabet = {p[1]:len(p[2]) for p in comm.productions}
comm.costs = {a:(lambda x: 1 + sum(x)) for a in comm.alphabet}
comm.EnumerateStrings(30)

(synth-fun countSketch ((x (BitVec 8))) (BitVec 8)
     (
	     (Start (BitVec 8) ( x
	     	     	       	 (let ((tmp (BitVec 8) Start) (m (BitVec 8) ConstBV) (n (BitVec 8) ConstBV))
                                  (bvadd (bvand tmp m) (bvand (bvlshr tmp n) m))
                                  )
                
		)
	     )
	     (ConstBV (BitVec 8) (
	       #x00 #xAA #xBB #xCC #xDD #xEE #xFF #xA0 #xB0 #xC0 #xD0 #xE0 #xF0 #x01 #x02 #x04
    		 )
	     
	     )
	   )
)



#changes to alg:
#H initialized to Min tree plus all trees with production diversions at the root
#need to track used elts of H since those can't be popped back into H
#need to find min elt in heap but also look up in set.
