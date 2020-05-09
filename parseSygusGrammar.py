import re
import cfg
import copy
import os
import time

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

#reads all grammars and writes enumeration results of form:
#filename (k time min_cost max_cost mean_cost) (k2 ...) ... 
def runAllFiles(kValues):
    numGoodFiles = 0
    for (root, _, files) in os.walk("./benchmarks"):
        for f in files:
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
                    print filename
                    g = parsefile(filename)
                    if not g:
                        print filename, " doesnt specify grammar."
                        continue
                    g.makeConsistent()
                    for k in kValues:
                        funcTime = time.time() #finds runtime of enumerate strings
                        trees = g.EnumerateStrings(k)
                        funcTime = time.time() - funcTime
                        minCost = float('inf')
                        maxCost = float('-inf')
                        sumCost = 0.0
                        for tree in trees[g.root]:
                            minCost = min(minCost, tree.cost)
                            maxCost = max(maxCost, tree.cost)
                            sumCost+= tree.cost
                        writeFile = open("enumerationTests.txt", "a")
                        writeFile.write("(" + str(k) + " " + str(round(funcTime, 3)) + " " + str(minCost) + " " + str(maxCost) + " " + str(round(sumCost / float(len(trees[g.root])), 3)) + ") ")
                        writeFile.close()
                    writeFile = open("enumerationTests.txt", "a")
                    writeFile.write(str(len(g.nonterminals)) + " " + str(len(g.productions)) + " " + str(len(g.alphabet)) + " " + str(max(g.alphabet.values())) + " " + str(len([a for a in g.alphabet if g.alphabet[a] == 0])) + " " + filename + "\n")
                    writeFile.close()
                    

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
