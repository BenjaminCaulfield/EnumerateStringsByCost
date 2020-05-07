import matplotlib.pyplot as plt
import cfg, time
############################ SYGUS TESTS #########################
##
############ Test Grammar 3 #############
###multiple nonterminals, uses max cost function
##tg5 = cfg.CFG()
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
nand = cfg.CFG()
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
iteg = cfg.CFG()
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
comm = cfg.CFG()
comm.root = "Start"
comm.nonterminals = ["Start"]
comm.productions = [["Start", "+", ["Start", "Start"]],
                    ["Start", "-", ["Start", "Start"]],
                    ["Start", "x", []],
                    ["Start", "y", []]]
comm.alphabet = {p[1]:len(p[2]) for p in comm.productions}
comm.costs = {a:(lambda x: 1 + sum(x)) for a in comm.alphabet}
#comm.EnumerateStrings(30)




comm = cfg.CFG()
comm.root = "Start"
comm.nonterminals = ["Start"]
comm.productions = [["Start", "+", ["Start", "Start"]],
                    ["Start", "-", ["Start", "Start"]],
                    ["Start", "x", []],
                    ["Start", "y", []]]
comm.alphabet = {p[1]:len(p[2]) for p in comm.productions}
comm.costs = {a:(lambda x: 1 + sum(x)) for a in comm.alphabet}
#comm.EnumerateStrings(30)



## Can't get around the let statement
##(synth-fun countSketch ((x (BitVec 8))) (BitVec 8)
##     (
##	     (Start (BitVec 8) ( x
##	     	     	       	 (let ((tmp (BitVec 8) Start) (m (BitVec 8) ConstBV) (n (BitVec 8) ConstBV))
##                                  (bvadd (bvand tmp m) (bvand (bvlshr tmp n) m))
##                                  )
##                
##		)
##	     )
##	     (ConstBV (BitVec 8) (
##	       #x00 #xAA #xBB #xCC #xDD #xEE #xFF #xA0 #xB0 #xC0 #xD0 #xE0 #xF0 #x01 #x02 #x04
##    		 )
##	     
##	     )
##)
##)



########### RUNNING TESTS ################

#Runs EnumerateStrings on different values of k for a fixed grammar.
#incremens size of k by inc until maxK is reached
#returns list of pairs (k, t) for runtime t and size k
#write results in form k t to given file
def testRuntime(g, maxK, inc, filename = "testresults.txt"):
    runtimes = []
    k = inc
    while k <= maxK:
        t = time.time()
        g.EnumerateStrings(k)
        t = time.time() - t
        runtimes.append((k, t))
        k += inc
    with open(filename, 'w') as f:
        for item in runtimes:
            f.write(str(item[0]) + " " + str(item[1]) + "\n")
    return runtimes
    
