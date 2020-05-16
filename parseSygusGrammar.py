import re
import cfg
import copy
import os
import time
import numpy
import random

def push(obj, l, depth):
   # print str(obj)
    
    while depth:
        l = l[-1]
        depth -= 1

    if(type(obj) == list):
        l.append(obj)
    elif obj == " ":
        l.append("")
    else:
        l[-1] += obj
def parse_parentheses(inStr):
    s = inStr
    assert s.find("(") != -1

    s = re.sub(";.*\n", "\n", s) #removes comments (everthing following ";" on a line)

    #removes white space around ( and )
    s = re.sub("\s*\(\s*","(", s)
    s = re.sub("\s*\)\s*",")", s)
    s = re.sub("\)",") ", s) #for some reason, needs a space after ')' to parse. 
    s = s[s.find("("):] #removes all characters before first '('
    #print s
    groups = []
    depth = 0

    try:
        inString = None
        for char in s:
            if char == '"' and not inString:
                inString = '"'
            elif char == '"':
                push(inString+'"', groups, depth)
                inString = None
            elif inString:
                inString += char
            elif char == '(':
                push([""], groups, depth)
                depth += 1
            elif char == ')':
                depth -= 1
            else:
                push(char, groups, depth)
    except IndexError:
        raise ValueError('Parentheses mismatch')

    if depth > 0:
        raise ValueError('Parentheses mismatch')
    else:
        return groups

#removes '' and '\n' from lists
def removeExcess(l):
    if type(l) == str:
        return l.replace("\n", "").replace("\t", "")
    l2 = [removeExcess(sub) for sub in l if not (sub == "" or sub == "\n" or sub == "\t")]
    return l2

randvals = [1757, 3909, 2700, 1631, 3039, 4062, 290, 3374, 3068, 3625, 2317, 187, 21, 2477, 2781, 869, 1516, 769, 2739, 3957, 1443, 1101, 1610, 2416, 3822, 2957, 4007, 270, 3748, 3809, 2537, 2655, 3473, 3063, 3750, 619, 2240, 878, 2078, 2587, 2655, 3464, 2134, 3802, 2153, 3955, 1171, 4199, 446, 3578, 660, 2219, 1804, 1845, 253, 3334, 2475, 994, 3296, 659, 605, 3737, 1195, 243, 4196, 3545, 858, 3466, 2150, 2293, 45, 3059, 107, 1422, 3970, 940, 43, 4145, 3522, 3552, 1334, 4063, 3066, 31, 4009, 64, 2415, 1523, 223, 4150, 890, 31, 2813, 3077, 3673, 4008, 2631, 3002, 1709, 435, 531, 3872, 3736, 1755, 3010, 2986, 1887, 2766, 382, 3126, 239, 3202, 3951, 2821, 2631, 229, 3396, 885, 688, 328, 64, 1446, 3628, 552, 3697, 1083, 223, 3904, 2905, 2856, 233, 3349, 577, 2780, 1781, 1140, 1006, 2278, 1131, 1279, 2825, 1280, 1266, 1237, 659, 1809, 2643, 417, 1201, 859, 746, 2534, 1636, 3059, 3184, 1297, 2408, 1419, 153, 4107, 3194, 3457, 2784, 1125, 4096, 3415, 1371, 1869, 3544, 570, 2224, 3334, 1229, 2195, 3049, 2815, 453, 1980, 2210, 295, 3857, 4125, 297, 3069, 3276, 619, 1454, 1522, 4011, 928, 636, 3115, 1085, 706, 4184, 1350, 274, 3953, 3575, 526, 1899, 4161, 1906, 3958, 2956, 784, 2936, 2348, 3842, 467, 2652, 3056, 2318, 591, 2515, 1830, 532, 231, 452, 3810, 2, 4045, 3864, 741, 2204, 2426, 3117, 717, 526, 1242, 3430, 3460, 4034, 603, 2577, 3214, 790, 1086, 2693, 812, 1829, 1606, 4156, 1676, 1044, 1957, 4209, 3384, 3254, 1036, 57, 1837, 2066, 891, 689, 2535, 3236, 2139, 1935, 621, 2426, 2409, 3854, 1023, 2597, 3861, 3417, 908, 1524, 2546, 563, 2809, 2896, 2136, 2447, 120, 4043, 4038, 3386, 1971, 3117, 889, 1836, 3118, 3608, 2138, 1626, 2290, 1644, 3989, 1059, 1244, 2825, 1771, 3798, 58, 711, 3434, 991, 3470]


#reads all grammars and writes enumeration results of form:
#filename (k time min_cost max_cost mean_cost) (k2 ...) ...
#also writes just runtimes to allGrammarTests_k
#returns list of function names and average times, sorted by times.
def runAllFiles(k):
    runtimes = []
    numRepeats = 5 #number of times enumerate is repeated for a single file
    for (root, _, files) in os.walk("./usedGrammars"):# replace with os.walk("./benchmarks") to run through all files
        for f in files:
            if f.endswith(".sl"):
                filename = os.path.join(root,f)
                afile = open(filename,mode='r')
                filestring = afile.read()
                afile.close()
                if re.search("synth-fun", filestring) and not re.search("(define-fun.*synth-fun)|(synth-fun.*let)", filestring, re.DOTALL):
                    g = parsefile(filename)
                    if not g:
                        assert False
                        continue
                    g.makeConsistent()
                    g.setCostsToDepth() #change to g.setCostsToSize() to get min-size terms
                    simpleWrite = open("allGrammarTests_depth_" + str(k) + ".txt", "a") #name of the file that just stores runtimes 
                    tests = []
                    for _ in range(0,numRepeats):
                        funcTime = time.time() #finds runtime of enumerate strings
                        trees = g.EnumerateStrings(k)
                        funcTime = time.time() - funcTime
                        tests.append(funcTime)
                    
                    runtimes.append((numpy.mean(tests), f))
                    minCost = float('inf')
                    maxCost = float('-inf')
                    sumCost = 0.0
                    for tree in trees[g.root]:
                        minCost = min(minCost, tree.cost)
                        maxCost = max(maxCost, tree.cost)
                        sumCost+= tree.cost
                    writeFile = open("enumerationTests.txt", "a")
                    writeFile.write("(" + str(k) + " " + str(round(numpy.mean(tests), 3)) + " " + str(minCost) + " " + str(maxCost) + " " + str(round(sumCost / float(len(trees[g.root])), 3)) + ") ")
                    writeFile.close()
                    writeFile = open("enumerationTests.txt", "a")
                    writeFile.write(str(len(g.nonterminals)) + " " + str(len(g.productions)) + " " + str(len(g.alphabet)) + " " + str(max(g.alphabet.values())) + " " + str(len([a for a in g.alphabet if g.alphabet[a] == 0])) + " " + filename + "\n")
                    writeFile.close()

                    simpleWrite.write(str(numpy.mean(tests)) + "\n")
                    simpleWrite.close()
                    print f + ": " + str(numpy.mean(tests))

    runtimes.sort()
    return runtimes
#    goodfile.close()

                    

#returns a cfg represented the relevant sygus grammar
def parsefile(filename):
    # Open a file: file
    afile = open(filename,mode='r')
 
    # read all lines at once
    filestring = afile.read()
 
    # close the file
    afile.close()

    return parseString(filestring)


def parseString(filestring):
    parsed_file = parse_parentheses(filestring)
    parsed_file = removeExcess(parsed_file)

    #print str(parsed_file)
    defined_funcs = []
    synth_pos = -1
    for x in range(0,len(parsed_file)):
        if type(parsed_file[x]) == list:
            if parsed_file[x][0] == "synth-fun":
                synth_pos = x
                break
            if parsed_file[x][0] == "define-fun":
                defined_funcs.append(x)
    assert synth_pos != -1

    #still need to handle defined funcs

    synthG = cfg.CFG()
    synthfunc = parsed_file[synth_pos]
    if len(synthfunc) < 5: #grammar isn't specified
        return None
    synthG.root = synthfunc[4][0][0]
    synthG.nonterminals = []# [synthG.root]
    for nontDef in synthfunc[4]:
        synthG.nonterminals.append(nontDef[0])
    for nontDef in synthfunc[4]:
        #synthG.nonterminals.append(nontDef[0])
        for prod in nontDef[2]:
            #if production is just defining a typed constant / variable
            if type(prod) == list and len(prod) == 2 and prod[1] == nontDef[1]:
                synthG.alphabet[prod[0]] = 0
                synthG.productions.append([nontDef[0], prod[0], []])
            #if production is a const or a single nonterminal
            elif type(prod) == str:
                #assert not prod in synthG.nonterminals, "A -> B production"

                #production of form A -> B
                if prod in synthG.nonterminals: #take RHS of B and append it to end of A
                    B_rhs = -1
                    for x in synthfunc[4]:
                        if x[0] == prod:
                            B_rhs = x[2]
                    assert B_rhs != -1

                    B_rhs = copy.copy(B_rhs)
                    nontDef[2] += B_rhs
                    
                else:
                    synthG.alphabet[prod] = 0
                    synthG.productions.append([nontDef[0], prod, []])
            #if production includes function calls
            elif type(prod) == list:
                assert len(prod) > 1 and type(prod[0]) == str
                if not prod[0] in synthG.alphabet:
                    synthG.alphabet[prod[0]] = len(prod) - 1 #, "Multiple arities found in " + str(prod[0])
                else:
                    #print str(prod)
                    assert synthG.alphabet[prod[0]] == len(prod) - 1, "Multiple arities found in '" + str(prod[0]) + "'"
                for p in prod[1:]:
                    #rhs is nested function. e.g., S -> g(g(A))
                    #Adds new nonterminal B and yield S -> g(B) , B -> g(A)
                    if type(p) == list:
                        #newNT = nontDef[0] + "->" + str(prod)
                        newNT = "[X[X" + nontDef[0] + str(p) + "X]X]"
                        synthG.nonterminals.append(newNT)
                        synthfunc[4].append([newNT, "Unknown", [p]]) #This will allow for multiple nested functions #NEED TO FIND WAY TO FIND TYPE
                        p = newNT #when production is created, newNT will be in production
                        #assert False, "handle nested production " + str(prod)
                    #rhs is nontermina or terminal
                    elif type(p) == str:
                        if p not in synthG.nonterminals:
                            if p in synthG.alphabet:
                                assert synthG.alphabet[p] == 0
                            synthG.alphabet[p] = 0
                            if p[0].isupper():
                                print "uppercase terminal '" + p + "' in " + str(prod)
                synthG.productions.append([nontDef[0], prod[0], [p for p in prod[1:]]])
            else:
                assert False, "no production type matches " + str(prod)
    #synthG.nonterminals = list(set(synthG.nonterminals)) #removes duplicates
    return synthG
    #return parsed_file
    
            
    

s = """(synth-fun SC ((s (BitVec 4)) (t (BitVec 4))) Bool
 ((Start Bool (
     true
     false
     (not Start)
     (and Start Start)
     (or Start Start)
     (= StartBv StartBv)
     (bvult StartBv StartBv)
     (bvslt StartBv StartBv)
     (bvuge StartBv StartBv)
     (bvsge StartBv StartBv)
   ))
   (StartBv (BitVec 4) (
     s
     t
     #x0
     #x8
     #x7
     (bvneg  StartBv)
     (bvnot  StartBv)
     (bvadd  StartBv StartBv)
     (bvsub  StartBv StartBv)
     (bvand  StartBv StartBv)
     (bvlshr StartBv StartBv)
     (bvor   StartBv StartBv)
     (bvshl  StartBv StartBv)
   ))
))"""

defstring = """
(define-fun qm ((a Int) (b Int)) Int
      (ite (< a 0) b a))

(synth-fun qm-foo ((x Int) (y Int) (ax Int) (ay Int) (bx Int) (by Int)) Int
    ((Start Int (x
                 y
                 ax
                 ay
                 bx
                 by
                 0
                 1
                 (- Start Start)
                 (+ Start Start)
                 (qm Start Start)))))
"""


stringdef = """
(synth-fun f ((name String)) String

    ((Start String (ntString))

     (ntString String (name "+" "-" "." ")" ")"

                       (str.++ ntString ntString)

                       (str.replace ntString ntString ntString)

                       (str.at ntString ntInt)

                       (int.to.str ntInt)

                       (str.substr ntString ntInt ntInt)))

      (ntInt Int (0 1 2 3 4 5

                  (+ ntInt ntInt)

                  (- ntInt ntInt)

                  (str.len ntString)

                  (str.to.int ntString)

                  (str.indexof ntString ntString ntInt)))

      (ntBool Bool (true false

                    (str.prefixof ntString ntString)

                    (str.suffixof ntString ntString)

                    (str.contains ntString ntString)))))

"""

def getDistinctGrammars():
    i = 0
    seenGrammars = []
    filenames = set() #files that have been written to new folder
    for (root, _, files) in os.walk("./benchmarks"):
        for f in files:
            #if i >= 3:
#                return
            if f in filenames:
                continue
            if f.endswith(".sl"):
                filename = os.path.join(root,f)
                afile = open(filename,mode='r')
                filestring = afile.read()
                afile.close()

#                p = filestring.find("synth-fun")
#                if not p:
#                    continue
                #print filestring
                #return
                if re.search("synth-fun", filestring) and not re.search("(define-fun.*synth-fun)|(synth-fun.*let)", filestring, re.DOTALL):
                    #print filename
                    g = parsefile(filename)
                    if not g:
                        #print filename, " doesnt specify grammar."
                        continue
                    g.makeConsistent()
                    if g not in seenGrammars:
                        seenGrammars.append(g)
                        newFile = open("./usedGrammars/" + f, 'w+')
                        newFile.write(filestring)
                        newFile.close()
                        filenames.add(f)
                        i += 1
                

randvals = [1757, 3909, 2700, 1631, 3039, 4062, 290, 3374, 3068, 3625, 2317, 187, 21, 2477, 2781, 869, 1516, 769, 2739, 3957, 1443, 1101, 1610, 2416, 3822, 2957, 4007, 270, 3748, 3809, 2537, 2655, 3473, 3063, 3750, 619, 2240, 878, 2078, 2587, 2655, 3464, 2134, 3802, 2153, 3955, 1171, 4199, 446, 3578, 660, 2219, 1804, 1845, 253, 3334, 2475, 994, 3296, 659, 605, 3737, 1195, 243, 4196, 3545, 858, 3466, 2150, 2293, 45, 3059, 107, 1422, 3970, 940, 43, 4145, 3522, 3552, 1334, 4063, 3066, 31, 4009, 64, 2415, 1523, 223, 4150, 890, 31, 2813, 3077, 3673, 4008, 2631, 3002, 1709, 435, 531, 3872, 3736, 1755, 3010, 2986, 1887, 2766, 382, 3126, 239, 3202, 3951, 2821, 2631, 229, 3396, 885, 688, 328, 64, 1446, 3628, 552, 3697, 1083, 223, 3904, 2905, 2856, 233, 3349, 577, 2780, 1781, 1140, 1006, 2278, 1131, 1279, 2825, 1280, 1266, 1237, 659, 1809, 2643, 417, 1201, 859, 746, 2534, 1636, 3059, 3184, 1297, 2408, 1419, 153, 4107, 3194, 3457, 2784, 1125, 4096, 3415, 1371, 1869, 3544, 570, 2224, 3334, 1229, 2195, 3049, 2815, 453, 1980, 2210, 295, 3857, 4125, 297, 3069, 3276, 619, 1454, 1522, 4011, 928, 636, 3115, 1085, 706, 4184, 1350, 274, 3953, 3575, 526, 1899, 4161, 1906, 3958, 2956, 784, 2936, 2348, 3842, 467, 2652, 3056, 2318, 591, 2515, 1830, 532, 231, 452, 3810, 2, 4045, 3864, 741, 2204, 2426, 3117, 717, 526, 1242, 3430, 3460, 4034, 603, 2577, 3214, 790, 1086, 2693, 812, 1829, 1606, 4156, 1676, 1044, 1957, 4209, 3384, 3254, 1036, 57, 1837, 2066, 891, 689, 2535, 3236, 2139, 1935, 621, 2426, 2409, 3854, 1023, 2597, 3861, 3417, 908, 1524, 2546, 563, 2809, 2896, 2136, 2447, 120, 4043, 4038, 3386, 1971, 3117, 889, 1836, 3118, 3608, 2138, 1626, 2290, 1644, 3989, 1059, 1244, 2825, 1771, 3798, 58, 711, 3434, 991, 3470]


#reads all grammars and writes enumeration results of form:
#filename (k time min_cost max_cost mean_cost) (k2 ...) ... 
def testEnumerationAccuracy():
    innerIndex = -1
    goodfile = open("goodfile", mode='r')
    goodCosts = eval(goodfile.read())
    goodfile.close()
    testfiles = []
    for (root, _, files) in os.walk("./testFiles"):
        for f in files:
            if f.endswith(".sl"):
                testfiles.append(os.path.join(root,f))
    testfiles.sort()
    testfiles.reverse()
    for filename in testfiles:
        afile = open(filename,mode='r')
        filestring = afile.read()
        afile.close()
        assert re.search("synth-fun", filestring) and not re.search("(define-fun.*synth-fun)|(synth-fun.*let)", filestring, re.DOTALL)
        print filename
        g = parsefile(filename)
        g.makeConsistent()
        g.setCostsToSize()
        assert g
        innerIndex += 1
        trees = g.EnumerateStrings(100)
        #writingFile = open("./testFiles/" + "0"*innerIndex + f, 'w+')
        #toWrite = open(filename)
        #writingFile.write(filestring)
        #writingFile.close()
        print filename
        costs = [x.cost for x in trees]
        costs.sort()
        assert costs == goodCosts[innerIndex], goodCosts[innerIndex] + ["SDFSDF"] + costs
                

#used for timing a single file. 
def timeFile(costs = "depth"):
    g = parsefile("./usedGrammars/phone-2-long-repeat.sl") #name of the file containing the grammar
    g.makeConsistent()
    #g.setCostsToDepth()
    g.setCostsToSize() #uncomment (and comment out line above) to set costs to size
    print g
    runtime = 0.0
    k = 1000 #starting k value
    maxTime = 5 #stops once runtime exceeds this value (in seconds)
    filename = open("phone_tests_"+costs, "a") #name of the file the results are written to
    numRepeats = 1 #times the test is repeated on the same k size before averaging
    while True: 
        tests = []
        while len(tests) < numRepeats:
            startTime = time.time()
            x = g.EnumerateStrings(k)
            runtime = time.time() - startTime
            tests.append(runtime)
        if numpy.mean(tests) >= maxTime: #ends once time is maxed out
            filename.write("-------------------------\n")
            filename.close()
            return
        writeString = str(k) + " " + str(round(numpy.mean(tests),4))
        print writeString
        filename.write(writeString + "\n")
        k += 10000 #increments by this amount each time
    filename.close()
    return
        

def testCaps():
    g = parsefile("./Hard_tests/phone-2.sl")
    g.makeConsistent()
    x = g.EnumerateStrings(100)
    for dt in x[g.root]:
        print dt


def testFunc(k):
    g = parsefile("./testFiles/0t4.sl")
    g.makeConsistent()
    g.setCostsToSize()
    print g
    runtime = time.time()
    x = g.EnumerateStrings(k)
    runtime = time.time() - runtime
    print k, runtime
    x.sort
    return x
    

#cfg1 = parsefile("testfile.sl")
#cfg2 = parsefile("testfile2.sl")
#cfg3 = parsefile("testA->B.sl")
#cfg4 = parsefile("nested_func_test.sl")



#funcname d[0][1]
#kth nonterminal section d[0][4][k]
#kth nonterminal name d[0][4][k][0]
#list of kth nonterminal prods d[0][4][k][2]

#DONE: remove ; comments 
#DONE: check if grammars are equivalent to avoid redundant grammars
#add nonterminals so that all productions use only one 
#handle def-fun before synth-func
#handle let statements
