#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/4/14
# @Author  : cjshao
# @File    : test.py
# @Software: PyCharm

from sklearn.ensemble import RandomForestClassifier
import math

domainlist = []

class Domain:
    def __init__(self, string, label=0):
        self.domainLength = len(string)
        self.letterEntropy = self.calEntropy(string)
        self.numbersCount = self.numbers(string)
        self.label = label

    def returnData(self):
        return [self.domainLength, self.letterEntropy, self.numbersCount]

    def returnLabel(self):
        if self.label == "notdga":
            return 0
        else:
            return 1

    def calEntropy(self, string):
        h = 0.0
        sumt = 0
        letter = [0] * 26
        string = string.lower()
        for i in range(len(string)):
            if string[i].isalpha():
                letter[ord(string[i]) - ord('a')] += 1
                sumt += 1
        for i in range(26):
            p = 1.0 * letter[i] / sumt
            if p > 0:
                h += -(p * math.log(p, 2))
        return h

    def numbers(self, string):
        sumt = 0
        for i in range(len(string)):
            if string[i].isdigit():
                sumt += 1
        return sumt



def initData(filename):
    with open(filename) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or line == "":
                continue
            tokens = line.split(",")
            domainName = tokens[0]
            # string = domainName.replace('.', '')
            label = tokens[1]
            domainlist.append(Domain(domainName, label))


def main():
    initData("train.txt")
    featureMatrix = []
    labelList = []
    for item in domainlist:
        featureMatrix.append(item.returnData())
        labelList.append(item.returnLabel())

    clf = RandomForestClassifier(random_state=0)
    clf.fit(featureMatrix, labelList)
    # print(clf.predict([Domain('google.dz').returnData(), Domain('g.dz').returnData()]))
    with open('test.txt') as f:
        filename = 'result.txt'
        with open(filename, 'w') as w:
            for line in f:
                line = line.strip()
                if line.startswith("#") or line == "":
                    continue
                oneResult = clf.predict([Domain(line).returnData()])[0]
                if oneResult == 0:
                    w.write(line+',notdga\n')
                else:
                    w.write(line+',dga\n')

if __name__ == '__main__':
    main()

