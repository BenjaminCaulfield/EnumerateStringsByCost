import cfg

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
g2 = cfg.CFG()
g2.nonterminals = ["S", "A"]
g2.alphabet = {"f":1, "a":0}
g2.productions = [["S", "f", ["A"]], ["A", "a", []], ["S", "a", []]]
g2.costs = {"f":(lambda x: x[0] + 1), "a":(lambda x: 10)}
assert g2.Knijkstra()[0]["S"]==10 #should equal 10
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
bvg = cfg.CFG()
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
g4 = cfg.CFG()
g4.root = "C"
g4.nonterminals = ["A", "B", "C"]
g4.alphabet = {"a":0, "b":2, "c":1, "d":3, "e":0, "f":2}
g4.productions = [["A", "a", []], ["A", "b", ["B", "C"]], ["B", "c", ["A"]], ["B", "d", ["A", "C", "A"]], ["C", "e", []], ["C", "f", ["B", "A"]]]
g4.costs = {"a":(lambda x: 4), "b":(lambda x: max(x[0], x[1])), "c":(lambda x: x[0] + 1), "d":(lambda x: x[0] + max(x[1],x[2])), "e":(lambda x: 9), "f":(lambda x: 0.5*(x[0] + x[1] + max(x[0], x[1])))}
#(mu4, prods4, order4) = g4.Knijkstra()
#trees4 = g4.getTreesFromProds(prods4, order4)
#####enums = g4.EnumerateStrings(3)
##
##


g5 = cfg.CFG()
g5.root = "S"
g5.nonterminals = ["S", "A"]
g5.alphabet = {"f":1, "g":2, "a":0}
g5.productions = [["S", "g", ["A", "A"]], ["A", "f", ["A"]], ["A", "a", []]]
x = g5.getDistFromRoot()
