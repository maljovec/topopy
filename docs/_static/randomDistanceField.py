from lxml.etree import *
from math import *
import numpy as np
import io


class randomField(object):

    def __init__(self,filename=None):

        if filename != None:
            self.fromfile(filename)

    def fromfile(self,filename):
        xml = parse(filename)
        field = xml.getroot()

        if field.tag == "DistanceField":
            self.field = randomDistanceField(field)
        elif field.tag == "InverseDistanceField":
            self.field = randomInverseDistanceField(field)
        else:
            raise ValueError("Field type not recognized")

    def eval(self,p,norm=None):
        return self.field.eval(p,norm)

    def dimension(self):
        return self.field.dimension



class randomDistanceField(object):

    def __init__(self,field=None):

        if field != None:
            self.fromxml(field)


    def fromfile(self,filename):
        xml = parse(filename)
        field = xml.getroot()

        self.fromxml(field)

    def fromxml(self,root):

        self.dimension = int(root.get('dimension'))
        size = int(root.get('size'))

        self.centers = []

        for p in list(root):
            coords = p.find('Coords')

            c = np.zeros(self.dimension,dtype=np.float64)
            for i in range(0,self.dimension):
                c[i] = float(coords.get('x%d'%i))


            power = float(p.find('Power').text)

            m = io.StringIO(p.find('Covariance').text)
            cov = np.loadtxt(m)

            self.centers.append([c,cov,power])


    def eval(self,p,norm=None):

        dist = [pow(sqrt(np.dot(p-x[0],np.dot(x[1],(p-x[0])))),x[2]) for x in self.centers]

        if norm == None:
            return min(dist)
        else:
            return pow(sum([pow(d,norm) for d in dist]),1./norm)


class randomInverseDistanceField(object):

    def __init__(self,field=None):

        if field != None:
            self.fromxml(field)


    def fromfile(self,filename):
        xml = parse(filename)
        field = xml.getroot()

        self.fromxml(field)

    def fromxml(self,root):

        self.dimension = int(root.get('dimension'))
        size = int(root.get('size'))

        self.centers = []

        for p in list(root):
            coords = p.find('Coords')

            c = np.zeros(self.dimension,dtype=np.float64)
            for i in range(0,self.dimension):
                c[i] = float(coords.get('x%d'%i))


            power = float(p.find('Power').text)
            value = float(p.find('Value').text)

            m = io.StringIO(p.find('Covariance').text)
            cov = np.loadtxt(m)

            self.centers.append([c,cov,power,value])


    def eval(self,p,norm=None):

        vals = [x[3] + pow(sqrt(np.dot(p-x[0],np.dot(x[1],(p-x[0])))),x[2]) for x in self.centers]
        weights = [pow(np.linalg.norm(p-x[0]) + 1e-10,norm) for x in self.centers]

        vals = [w*v for w,v in zip(weights,vals)]

        return sum(vals) / sum(weights)

if __name__ == '__main__':

    from sys import argv

    field = randomDistanceField(argv[1])

    for i in range(1,10):
        print(field.eval([i,i,i]))


