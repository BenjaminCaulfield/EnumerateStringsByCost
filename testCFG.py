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


randVals = [1757, 3909, 2700, 1631, 3039, 4062, 290, 3374, 3068, 3625, 2317, 187, 21, 2477, 2781, 869, 1516, 769, 2739, 3957, 1443, 1101, 1610, 2416, 3822, 2957, 4007, 270, 3748, 3809, 2537, 2655, 3473, 3063, 3750, 619, 2240, 878, 2078, 2587, 2655, 3464, 2134, 3802, 2153, 3955, 1171, 4199, 446, 3578, 660, 2219, 1804, 1845, 253, 3334, 2475, 994, 3296, 659, 605, 3737, 1195, 243, 4196, 3545, 858, 3466, 2150, 2293, 45, 3059, 107, 1422, 3970, 940, 43, 4145, 3522, 3552, 1334, 4063, 3066, 31, 4009, 64, 2415, 1523, 223, 4150, 890, 31, 2813, 3077, 3673, 4008, 2631, 3002, 1709, 435, 531, 3872, 3736, 1755, 3010, 2986, 1887, 2766, 382, 3126, 239, 3202, 3951, 2821, 2631, 229, 3396, 885, 688, 328, 64, 1446, 3628, 552, 3697, 1083, 223, 3904, 2905, 2856, 233, 3349, 577, 2780, 1781, 1140, 1006, 2278, 1131, 1279, 2825, 1280, 1266, 1237, 659, 1809, 2643, 417, 1201, 859, 746, 2534, 1636, 3059, 3184, 1297, 2408, 1419, 153, 4107, 3194, 3457, 2784, 1125, 4096, 3415, 1371, 1869, 3544, 570, 2224, 3334, 1229, 2195, 3049, 2815, 453, 1980, 2210, 295, 3857, 4125, 297, 3069, 3276, 619, 1454, 1522, 4011, 928, 636, 3115, 1085, 706, 4184, 1350, 274, 3953, 3575, 526, 1899, 4161, 1906, 3958, 2956, 784, 2936, 2348, 3842, 467, 2652, 3056, 2318, 591, 2515, 1830, 532, 231, 452, 3810, 2, 4045, 3864, 741, 2204, 2426, 3117, 717, 526, 1242, 3430, 3460, 4034, 603, 2577, 3214, 790, 1086, 2693, 812, 1829, 1606, 4156, 1676, 1044, 1957, 4209, 3384, 3254, 1036, 57, 1837, 2066, 891, 689, 2535, 3236, 2139, 1935, 621, 2426, 2409, 3854, 1023, 2597, 3861, 3417, 908, 1524, 2546, 563, 2809, 2896, 2136, 2447, 120, 4043, 4038, 3386, 1971, 3117, 889, 1836, 3118, 3608, 2138, 1626, 2290, 1644, 3989, 1059, 1244, 2825, 1771, 3798, 58, 711, 3434, 991, 3470]
