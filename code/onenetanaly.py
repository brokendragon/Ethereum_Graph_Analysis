# -*- coding: utf-8 -*-
#! /usr/bin/env python
# test.py
import networkx as nx
import matplotlib.pyplot as plt
import os
from math import sqrt
import numpy as np  
from numpy import linalg as la

#
#Aim to analysis the mfg, network has been merged by single edges.
#

def multipl(a,b):
	sumofab=0.0
	for i in range(len(a)):
		temp=a[i]*b[i]
		sumofab+=temp
	return sumofab
 
def corrcoef(y):
	x=range(0, len(y))
	n=len(x)
	#print x
	#print y
	#求和
	sum1=sum(x)
	sum2=sum(y)
    #求乘积之和
	sumofxy=multipl(x,y)
    #求平方和
	sumofx2 = sum([pow(i,2) for i in x])
	sumofy2 = sum([pow(j,2) for j in y])
	num=sumofxy-(float(sum1)*float(sum2)/n)
    #计算皮尔逊相关系数
	den=sqrt((sumofx2-float(sum1**2)/n)*(sumofy2-float(sum2**2)/n))
	return num/den

def norm(v):
  sum = float(0)
  for i in range(len(v)):
    sum += v[i]**2
  return sum**(0.5)

def cosSimilar(inB): 
	inA=range(0,len(inB))  
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
		for line in fg:
			count += 1
			if count == 1:
				continue
			info = line.split(',')
			fromaddr = info[0]
			toaddr = info[1]
			next = info[2].find('\n')
			if next <= 0:
				weight = float(info[2])
			else:
				weight = float(info[2][0:next])
			'''if weight.find(point_s) > 0:
				weight'''
			#print fromaddr+' '+toaddr+' '+str(weight)
			if fromaddr != toaddr:  #stop self cycle
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

accountinfo = {}
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
				pass
			accountinfo[info] = account
fl.close() 
print len(accountinfo)
point_s = '.'
path_s = '/home/bill/Desktop/end/one'
G = nx.DiGraph()
selfloop = []
#addnode(G)
files = os.listdir(path_s)
for file in files:
	addfile(path_s+"/"+file,G)

'''
ind = list(G.in_degree().values())
outd = list(G.out_degree().values())
print 'person indegree: ',corrcoef(ind,outd)
print 'cos indegree:    ',cosSimilar(ind,outd)
#print 'correlation:     ',nx.degree_pearson_correlation_coefficient(G)
#G = G.to_undirected()
#print nx.average_clustering(G)
#print 'correlation:     ',nx.degree_pearson_correlation_coefficient(G)

#print 'node num',nx.number_of_nodes(G)
with open('/home/bill/Desktop/one_selfloop','a')     as  f:
	print 'writing'
	for i in selfloop:
		f.write(i+'\n')
f.close()'''
'''
degree = nx.degree_histogram(G)
ind = in_degree_histogram(G) 
outd = out_degree_histogram(G)
print 'ok'

x = []
y = []
x = range(len(degree))                        
y = [z / float(sum(degree)) for z in degree]  
plt.loglog(x,y,color="blue",linewidth=2)        
plt.savefig("one_d.png")
plt.clf()
                                                    
x = []
y = []
x = range(len(ind))                        
y = [z / float(sum(ind)) for z in ind]  
plt.loglog(x,y,color="blue",linewidth=2)          
plt.savefig("one_ind.png")
plt.clf()
 
x = []
y = []
x = range(len(outd))                             
y = [z / float(sum(outd)) for z in outd]  
plt.loglog(x,y,color="blue",linewidth=2)          
plt.savefig("one_outd.png")
plt.clf() 
'''


'''print 'person indegree: '+str(corrcoef(ind))
print 'cos indegree:    '+str(cosSimilar(ind))
print 'person outdegree: '+str(corrcoef(outd))
print 'cos outdegree:    '+str(cosSimilar(outd))

smax = 0
wmax = 0
countl = 0
countll = 0
for i in nx.strongly_connected_components(G):#ret scc list  easy to get the largest one	
	countl+=1	
	if len(i) > smax:
		smax = len(i)
for i in nx.weakly_connected_components(G):#ret wcc list
	countll+=1
	if len(i) > wmax:
		wmax = len(i)
print 'strongly max: ',countl,smax
print 'weakly max:   ',countll,wmax
print 'correlation:     ',nx.degree_pearson_correlation_coefficient(G)#correlation'''
print 'ok'
'''#print 'node num',nx.number_of_nodes(G)
#print nx.transitivity(G)
print nx.degree_pearson_correlation_coefficient(G)
print nx.is_attracting_component(G)
print nx.number_attracting_components(G)
print nx.is_semiconnected(G)
print nx.flow_hierarchy(G)
G = G.to_undirected()
#print 'ok,now is undirect'
print nx.is_connected(G)
print nx.number_connected_components(G)
print nx.average_clustering(G)'''



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



def turntof(dic):
	plt.clf()
	x = []
	y = []	
	for (k,v) in dic.items():
		x.append(k)
		y.append(v)
	plt.plot(x,y, 'ro')

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

##PageRank!!!!!!
print '$PageRank'
page = []
pagerank = nx.pagerank(G)        #it will be very big
PRdic = {}
for key,value in pagerank.items():
	PRtop(PRdic,key,value)
	page.append(value)
PRdics = sorted(PRdic.iteritems(), key=lambda d:d[1], reverse = True)
for i in range(20):
	(k,v) = PRdics[i]
	print account.get(k),v

#print retdic(page)
pagedic = retdic(page)
pagedict= sorted(pagedic.iteritems(), key=lambda d:d[0], reverse = True)
drawdic(pagedict)
plt.savefig("one_new_pagerank.png")

'''#Cycles!!!!!
print '$Cycles'
cyclelist = []
for i in nx.simple_cycles(G):   #A generator of lists of cycles  not for twonet
	cyclelist.append(len(i))
#print retdic(cyclelist)
print 'break'
turntof(retdic(cyclelist))
plt.savefig("cycle.png")'''


#G = G.to_undirected()
print 'ok,now is undirect'
'''
#Clustering!!!!!
print '$Clustering'
clustering = nx.clustering(G)  
cluster = [] 
for key,value in clustering.items():
	cluster.append(value)
#print retdic(cluster) 
print 'break'     
turntof(retdic(cluster))
plt.savefig("clustering.png")


#Rich Club
print '$Rich Club'              #already get the list
richclub =  nx.rich_club_coefficient(G,normalized=True,Q=100)   #is not defined for directed graphs
#getRCcount(richclub)
rich = []
for key,value in richclub.items():
	rich.append(value)
#print richclub
turntof(richclub)
plt.savefig("richclub.png")

#betweenness_centrality
print '$betweenness_centrality'
bc = nx.betweenness_centrality(G)
with open('/home/bill/Desktop/betweenness_centrality','w') as f:
	for k,v in bc.items():
		f.write(k+','+v+'\n')	
f.close()
'''
