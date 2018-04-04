# -*- coding: utf-8 -*-
#! /usr/bin/env python
# test.py
import networkx as nx
import matplotlib.pyplot as plt
import os
import random
from math import sqrt
import numpy as np  
from numpy import linalg as la

#
#Aim to analysis the cig, network has been merged by single edges.
#

def getpearsonpvalue(p5,p6):
	dis_p = p5
	tau_p = p6
	p5_avg = float(sum(p5))/len(p5)
	p6_avg = float(sum(p6))/len(p6)
	p5[:] = [x - p5_avg for x in p5]
	p6[:] = [x - p6_avg for x in p6]
	p5 = sum(map(lambda (a,b):a*b, zip(p5,p6)))/sqrt(sum(map(lambda x : x*x, p5))*sum(map(lambda x : x*x, p6)))
	p = 0
	count = 10000
	for i in range(10000):
		dis1 = []
		tau1 = []
		dis1 = dis_p
		a = range(1,len(dis_p)+1)
		random.shuffle(a)
		both = sorted(zip(tau_p,a),key=lambda x:x[1])
		for k,v in both:
			tau1.append(k)
		dis1_avg = float(sum(dis1))/len(dis1)
		tau1_avg = float(sum(tau1))/len(tau1)
		dis1[:] = [x - dis1_avg for x in dis1]
		tau1[:] = [x - tau1_avg for x in tau1]
		dis1 = sum(map(lambda (a,b):a*b, zip(dis1,tau1)))/sqrt(sum(map(lambda x : x*x, dis1))*sum(map(lambda x : x*x, tau1)))
		if dis1 <= p5:
			p = p+1

	print p
	print float(p)/count

def multipl(a,b):
	sumofab=0.0
	for i in range(len(a)):
		temp=a[i]*b[i]
		sumofab+=temp
	return sumofab
 
def corrcoef(x,y):
	#print x
	#print y
	n=len(x)
	sum1=sum(x)
	sum2=sum(y)
	sumofxy=multipl(x,y)
	sumofx2 = sum([pow(i,2) for i in x])
	sumofy2 = sum([pow(j,2) for j in y])
	num=sumofxy-(float(sum1)*float(sum2)/n)
	den=sqrt((sumofx2-float(sum1**2)/n)*(sumofy2-float(sum2**2)/n))
	return num/den

def norm(v):
  sum = float(0)
  for i in range(len(v)):
    sum += v[i]**2
  return sum**(0.5)

def cosSimilar(inA,inB): 
	#inA=range(0,len(inB))  
	denom=norm(inA)*norm(inB) 
	inA=np.mat(inA) 
	inB=np.mat(inB) 
	num=float(inA*inB.T) 
	return num/denom#缩放0.5+0.5*(num/denom)

def in_degree_histogram(G):
	inde = list(G.in_degree().values())
	dmax = max(inde)+1
	freq = [0 for d in range(dmax)]
	for d in inde:
		freq[d] += 1
	return freq

def out_degree_histogram(G):
	inde = list(G.out_degree().values())
	dmax = max(inde)+1
	freq = [0 for d in range(dmax)]
	for d in inde:
		freq[d] += 1
	return freq

def addfile(path,G):
	with open(path,'r') as fg:
		count = 0
		countl = 0
		for line in fg:
			count += 1
			if count == 1:
				continue
			info = line.split(',')
			fromaddr = info[0]
			toaddr = info[1]
			next = info[2].find('\n')
			if next <= 0:
				weight = int(info[2])
			else:
				weight = int(info[2][0:next])
			if fromaddr != toaddr:
				pass#selfloop.append(fromaddr)
			G.add_weighted_edges_from([(fromaddr,toaddr,weight)])	
	fg.close()

def addnode(G):
	with open('/home/bill/Desktop/allnodeinfo','r') as f:
		count = 0
		lcount = 0
		for line in f:
			if lcount == 0:
				lcount += 1			
				continue
			else:
				lcount += 1
				node = line.split(',')[0]
				weight = line.split(',')[1]
				nl = weight.find('\n')
				if nl > 0:
					weight = int(weight[0:nl])
				else:
					weight =int(weight)
				if weight == -1:
					G.add_node(node)
					count += 1
		print count
	f.close()

path_s = '/home/bill/Desktop/end/three'
G = nx.DiGraph()
selfloop = []
#addnode(G)
files = os.listdir(path_s)
for file in files:
	addfile(path_s+"/"+file,G)

'''degree = nx.degree_histogram(G)
ind = in_degree_histogram(G) 
outd = out_degree_histogram(G)'''
'''accountinfo = {}
with open('/home/bill/Desktop/newestlands') as fl:
	count = 0	
	for line in fl:
		if count == 0:
			count += 1			
			continue
		else:
			count += 1
			account = line.split(',')[0]  
			info = line.split(',')[1] 	
			if line.find('Sc') >= 0:
				#pass
				accountinfo[info] = account
fl.close() 
print len(accountinfo)'''

#G = G.to_undirected()
print 'ok,now is undirect'
#print nx.average_clustering(G) #jia


'''for i in range(len(ind)):
	if (ind[i]/float(sum(ind))) > 0.0001:
		if i>20:
			if i<50:
				print i,ind[i],(ind[i]/float(sum(ind)))'''

#gd = G.degree()
#gi = G.in_degree()
#go = G.out_degree()

#dl = sorted(gd.iteritems(), key=lambda d:d[1], reverse = True)
#indl = sorted(gi.iteritems(), key=lambda d:d[1], reverse = True)
#oudl = sorted(go.iteritems(), key=lambda d:d[1], reverse = True)
'''print 'dl'
for k,v in dl:
	count += 1
	if count>20:
		break
	print accountinfo.get(k),v
count = 0
print 'indl'
for k,v in indl:
	count += 1
	if count>20:
		break
	print accountinfo.get(k),v
count = 0
print 'oudl'
for k,v in oudl:
	count += 1
	if count>20:
		break
	print accountinfo.get(k),v'''
'''for k,v in gi.items():
	if k == address:
		print v
for k,v in go.items():
	if k == address:
		print v
'''
#print "selfloop: ",len(selfloop)
print 'start,now is direct'
#print 'node num',nx.number_of_nodes(G)

'''
ind = list(G.in_degree().values())
outd = list(G.out_degree().values())
inddic = G.in_degree()
outddic = G.out_degree()
for k,v in inddic.items():
	if accountinfo.get(k,0) == 0:
		del inddic[k]
for k,v in outddic.items():
	if accountinfo.get(k,0) == 0:
		del outddic[k]
ind = list(inddic.values())
outd = list(outddic.values())

dlist = map(lambda (a,b):(a,b),zip(ind,outd))
with open('/home/bill/Desktop/new/dlist','w') as f:
	for k,v in dlist:
		f.write(str(k)+','+str(v)+'\n')
f.close()
print 'finish'
#getpearsonpvalue(ind,outd)
#print 'person indegree: ',corrcoef(ind,outd)
#print 'cos indegree:    ',cosSimilar(ind,outd)
print 'correlation:     ',nx.degree_pearson_correlation_coefficient(G)'''
'''G = G.to_undirected()
print 'clustering:  ',nx.average_clustering(G)'''
'''with open('/home/bill/Desktop/three_selfloop','a')     as  f:
	print 'writing'
	for i in selfloop:
		f.write(i+'\n')
f.close()'''
'''degree = nx.degree_histogram(G)
ind = in_degree_histogram(G) 
outd = out_degree_histogram(G)
print 'ok'

x = []
y = []
x = range(len(degree))                          
y = [z / float(sum(degree)) for z in degree]  
plt.loglog(x,y,color="blue",linewidth=2)        
plt.savefig("three_d.png")
plt.clf()
                                                  
x = []
y = []
x = range(len(ind))                       
y = [z / float(sum(ind)) for z in ind]  
plt.loglog(x,y,color="blue",linewidth=2)      
plt.savefig("three_ind.png")
plt.clf()
 
x = []
y = []
x = range(len(outd))                          
y = [z / float(sum(outd)) for z in outd]  
plt.loglog(x,y,color="blue",linewidth=2)         
plt.savefig("three_outd.png")
'''

'''print 'person indegree: '+str(corrcoef(ind))
print 'cos indegree:    '+str(cosSimilar(ind))
print 'person outdegree: '+str(corrcoef(outd))
print 'cos outdegree:    '+str(cosSimilar(outd))
'''
'''
smax = 0
wmax = 0
for i in nx.strongly_connected_components(G):
	if len(i) > smax:
		smax = len(i)
for i in nx.weakly_connected_components(G):
	if len(i) > wmax:
		wmax = len(i)
print 'strongly max: ',smax
print 'weakly max:   ',wmax
'''
'''#print 'correlation:     ',nx.degree_pearson_correlation_coefficient(G)

print nx.transitivity(G)  #jia
print nx.number_strongly_connected_components(G)
print nx.number_weakly_connected_components(G)
print nx.number_attracting_components(G)
print nx.flow_hierarchy(G)
#G = G.to_undirected()
#print 'ok,now is undirect'
#print nx.average_clustering(G) #jia
'''

def retdic(tlist):
	dic = {}
	for i in tlist:
		weight = dic.get(i,0)
		if weight == 0:
			dic[i] = 1
		else:
			dic[i] = weight + 1
	return dic

def drawdic(dic):
	plt.clf()
	x = []
	y = []	
	for l in range(len(dic)):
		(k,v) = dic[l]
		for i in range(v):
			y.append(k)
	x = range(len(y))
	plt.loglog(x,y, 'ro')

def writedic(kdic): #new add
	with open('/home/bill/Desktop/threerichclub','w') as f:
		for (k,v) in kdic.items():
			f.write(str(k)+','+str(v)+'\n')
	f.close()

def turntof(dic):
	plt.clf()
	x = []
	y = []	
	kdic = {}
	for (k,v) in dic.items():
		if v >= 1.0:
			kdic[k] = v
		x.append(k)   
		y.append(v)
	plt.plot(x,y, 'ro')
	writedic(kdic)

def PRtop(dic,node,value):
	if len(dic) < 20:
		dic[node] = value
	else:
		for (k,v) in dic.items():
			if value > v:
				del dic[k]
				dic[node] = value
				break


def getRCcount(dic):
	count1 = 0
	count2 = 0
	for (k,v) in dic.items():
		if v>0.8:
			count1 += 1
		elif v>0.6:
			count2 += 1
	print count1,count2

'''##PageRank!!!!!!
print '$PageRank'
page = []
pagerank = nx.pagerank(G)        #it will be very big
PRdic = {}
for key,value in pagerank.items():
	page.append(value)
#retdic(page)
pagedic = retdic(page)
pagedict= sorted(pagedic.iteritems(), key=lambda d:d[0], reverse = True)
drawdic(pagedict)
plt.savefig("three_new_pagerank.png")
PRlist = sorted(pagerank.iteritems(), key=lambda d:d[1], reverse = True)
count = 0
for k,v in PRlist:
	if count < 20:
		print accountinfo.get(k),v
		count += 1
	else:
		break
#Cycles!!!!!
print '$Cycles'
cyclelist = []
with open('/home/bill/Desktop/new/twosizecycle', 'w') as f:
	for i in nx.simple_cycles(G):   #A generator of lists of cycles  
		#cyclelist.append(len(i))
		if len(i) == 2:
			f.write(accountinfo.get(i[0])+'->'+accountinfo.get(i[1])+'\n') 
f.close()
#print retdic(cyclelist)
#turntof(retdic(cyclelist))
#plt.savefig("cycle.png")
'''

#G = G.to_undirected()
print 'ok,now is undirect'

'''#Clustering!!!!!
print '$Clustering'
clustering = nx.clustering(G)  
cluster = [] 
for key,value in clustering.items():
	cluster.append(value)
#print retdic(cluster)      
turntof(retdic(cluster))
plt.savefig("clustering.png")


#Rich Club
print '$Rich Club'              #already get the list
richclub =  nx.rich_club_coefficient(G,normalized=False)   #is not defined for directed graphs
#getRCcount(richclub)
count = 0
for k,v in richclub.items():
	if v>0.8:
		count += 1
print count 
#print richclub
#turntof(richclub)
#plt.savefig("richclub.png")
'''
