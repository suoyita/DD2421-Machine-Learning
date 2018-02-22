import sys

sys.path.append('/Users/chengkai/Desktop/coding/Homework/KTH/Machine_Learning/dectrees/python')

import dtree
import monkdata as m
import drawtree_qt5 as draw
import random

# "assignment 1"
print('Entropy:')
print(dtree.entropy(m.monk1))
print(dtree.entropy(m.monk2))
print(dtree.entropy(m.monk3))
print('\n')

# "assignment 3"
print('Information Gain:')
print('monk1:')
for i in range(6):
    print('attribute', i + 1, ':', dtree.averageGain(m.monk1, m.attributes[i]))
print('monk2:')
for i in range(6):
    print('attribute', i + 1, ':', dtree.averageGain(m.monk2, m.attributes[i]))
print('monk3:')
for i in range(6):
    print('attribute', i + 1, ':', dtree.averageGain(m.monk3, m.attributes[i]))
print('\n')

# assignment 5
t1 = dtree.buildTree(m.monk1, m.attributes)
print(dtree.check(t1, m.monk1test))
#draw.drawTree(t1)
t2 = dtree.buildTree(m.monk2, m.attributes)
print(dtree.check(t2, m.monk2test))
t3 = dtree.buildTree(m.monk3, m.attributes)
print(dtree.check(t3, m.monk3test))
print('\n')


# assignment 7
def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]


fraction = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
for i in fraction:
    sumScore = 0
    minScore = 1
    maxScore = 0
    for j in range(100):
        score = 0
        bestScore = 0

        monk1train, monk1val = partition(m.monk1, i)
        t = dtree.buildTree(monk1train, m.attributes)
        bestTree = t
        bestScore = dtree.check(t, monk1val)
        while True:
            dt = dtree.allPruned(bestTree)
            lastBestScore = bestScore
            for ts in dt:
                score = dtree.check(ts, monk1val)
                if score > bestScore:
                    bestScore = score
                    bestTree = ts
            if bestScore == lastBestScore:
                break
        bestScore = dtree.check(bestTree, m.monk1test)
        sumScore += bestScore
        minScore = min(minScore, bestScore)
        maxScore = max(maxScore, bestScore)

    meanScore = sumScore / 100
    print('fraction:', i, ' score:', meanScore, 'minScore:', minScore, 'maxScore:',maxScore)

print('\n')

fraction = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
for i in fraction:
    sumScore = 0
    minScore = 1
    maxScore = 0
    for j in range(100):
        score = 0
        bestScore = 0

        monk3train, monk3val = partition(m.monk3, i)
        t = dtree.buildTree(monk3train, m.attributes)
        bestTree = t
        bestScore = dtree.check(t, monk3val)
        while True:
            dt = dtree.allPruned(bestTree)
            lastBestScore = bestScore
            for ts in dt:
                score = dtree.check(ts, monk3val)
                if score > bestScore:
                    bestScore = score
                    bestTree = ts
            if bestScore == lastBestScore:
                break
        bestScore = dtree.check(bestTree, m.monk3test)
        sumScore += bestScore
        minScore = min(minScore, bestScore)
        maxScore = max(maxScore, bestScore)

    meanScore = sumScore / 100
    print('fraction:', i, ' score:', meanScore, 'minScore:', minScore, 'maxScore:',maxScore)
