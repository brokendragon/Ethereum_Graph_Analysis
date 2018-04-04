# -*- coding: utf-8 -*-
#! /usr/bin/env python
# test.py
import networkx as nx
import matplotlib.pyplot as plt
import os
from math import sqrt
import numpy as np  
from numpy import linalg as la

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
		edgecount = 0
		count = 0
		for line in fg:
			count += 1
			if count == 1:
				continue
			info = line.split(',')
			#print info
			fromaddr = info[0]
			toaddr = info[1][0:8]
			#print fromaddr+' '+toaddr
			weight = 1
			if fromaddr != toaddr:
				#pass#selfloop.append(fromaddr)
				G.add_weighted_edges_from([(fromaddr,toaddr,weight)])
	#print edgecount
	fg.close()

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
				pass
			accountinfo[info] = account
fl.close() 
print len(accountinfo)'''

path_s = '/home/bill/Desktop/end/two'
G = nx.DiGraph()
files = os.listdir(path_s)
for file in files:
	addfile(path_s+"/"+file,G)
print nx.number_of_nodes(G)
'''G = G.to_undirected()
print nx.average_clustering(G)
#print 'correlation:     ',nx.degree_pearson_correlation_coefficient(G)
print 'start,now is direct'''
'''degree = nx.degree_histogram(G)
ind = in_degree_histogram(G) 
outd = out_degree_histogram(G)






x = []
y = []
x = range(len(degree))                           
y = [z / float(sum(degree)) for z in degree]  ）
plt.loglog(x,y,color="blue",linewidth=2)       
plt.savefig("two_d.png")
plt.clf()
 

x = []
y = []
x = range(len(ind))                          
y = [z / float(sum(ind)) for z in ind]  	
#plt.loglog(x,y,color="blue",linewidth=2,'ro')      
plt.plot(x,y,'ro',color="blue",linewidth=2)  
plt.savefig("two_ind.png")
plt.clf()

x = []
y = []
x = range(len(outd))                    
y = [z / float(sum(outd)) for z in outd]  
plt.loglog(x,y,color="blue",linewidth=2)        
plt.savefig("two_outd.png")
plt.clf()
'''
'''
gd = G.degree()
gi = G.in_degree()
go = G.out_degree()
count = 0
dl = sorted(gd.iteritems(), key=lambda d:d[1], reverse = True)
indl = sorted(gi.iteritems(), key=lambda d:d[1], reverse = True)
oudl = sorted(go.iteritems(), key=lambda d:d[1], reverse = True)
print 'dl'
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
	print accountinfo.get(k),v
'''

'''print 'ok'
print 'person indegree: '+str(corrcoef(ind))
print 'cos indegree:    '+str(cosSimilar(ind))
print 'person outdegree: '+str(corrcoef(outd))
print 'cos outdegree:    '+str(cosSimilar(outd))
'''
#print nx.number_weakly_connected_components(G)
smax = 0
wmax = 0
for i in nx.strongly_connected_components(G):#ret scc list easy to get the largest one	
	if len(i) > smax:
		smax = len(i)
for i in nx.weakly_connected_components(G):#ret wcc list
	if len(i) > wmax:
		wmax = len(i)
print 'strongly max: ',smax
print 'weakly max:   ',wmax
'''
print nx.transitivity(G)  #jia
print nx.number_strongly_connected_components(G)
print nx.number_weakly_connected_components(G)
print nx.number_attracting_components(G)
print nx.flow_hierarchy(G)
#print 'correlation:     ',nx.degree_pearson_correlation_coefficient(G)

x = range(len(degree))                      
y = [z / float(sum(degree)) for z in degree]  
plt.loglog(x,y,color="blue",linewidth=2)           
plt.show()                                                    '''
'''


'''#degree distritribution
#print G.degree()               #it will be very big
#print nx.degree_histogram(G)   #it will be very big'''
'''#Clustering
#print nx.triangles(G) #triangles() is not defined for directed graphs.
print nx.transitivity(G)
#print nx.clustering(G)   #Clustering algorithms are not defined ', 'for directed graphs.
#print nx.average_clustering(G)   #Clustering algorithms are not defined ', 'for directed graphs.
print nx.square_clustering(G)'''

'''#Assortativity
#print nx.degree_assortativity_coefficient(G)     #ValueError: array is too big
#print nx.attribute_assortativity_coefficient(G, attribute)   #attribute need to define
#print nx.numeric_assortativity_coefficient(G, attribute)   #attribute need to define'''
###print nx.degree_pearson_correlation_coefficient(G)

'''#communities
#print nx.communicability(G)         #not implemented for directed type
#print nx.communicability_exp(G)      #not implemented for directed type
#print nx.communicability_centrality(G)     #not implemented for directed type
#print nx.communicability_centrality_exp(G)   # not implemented for directed type
#print nx.communicability_betweenness_centrality(G)   # not implemented for directed type
#print nx.estrada_index(G)   # not implemented for directed type'''

#components!!!!
'''##Connectivity
#print nx.is_connected(G)                    #For undirected graphs only.
#print nx.number_connected_components(G)      #For undirected graphs only.
#print nx.connected_components(G)            #For undirected graphs only.
#print nx.connected_component_subgraphs(G)    #For undirected graphs only.
#print nx.node_connected_component(G,n)     #For undirected graphs only.'''
##Strong connectivity                                     #problem of object has get over
###print nx.is_strongly_connected(G)     
###print nx.number_strongly_connected_components(G)  
'''##print nx.strongly_connected_components(G)         #A generator of sets of nodes
#for i in nx.strongly_connected_components(G):     #can print
#	print i
##print nx.strongly_connected_component_subgraphs(G)     #A generator of sets of subgraphs  
#for i in nx.strongly_connected_component_subgraphs(G):  #can draw
#	nx.draw(i)
#	plt.show()
#print nx.strongly_connected_components_recursive(G)    #Recursive version,can use like strongly_connected_components
#print nx.kosaraju_strongly_connected_components(G)     #Can use like strongly_connected_components
#for i in nx.kosaraju_strongly_connected_components(G):
#	print i
#print nx.condensation(G)                              #can use to draw
nx.draw(nx.condensation(G))                         # can draw the condensation graph
plt.savefig("ba.png")
plt.show()'''
##Weak connectivity
###print nx.is_weakly_connected(G)
###print nx.number_weakly_connected_components(G)
'''#print nx.weakly_connected_components(G)              #A generator of sets of nodes 
#print nx.weakly_connected_component_subgraphs(G)     #A generator of sets of subgraphs'''
##Attracting components
###print nx.is_attracting_component(G)
###print nx.number_attracting_components(G)
'''#print nx.attracting_components(G)           #A generator of sets of nodes
#print nx.attracting_component_subgraphs(G)  #A generator of sets of subgraphs'''
'''##Biconnected components
#print nx.is_biconnected(G)           #not implemented for directed type
#print nx.biconnected_components(G)   #not implemented for directed type
#print nx.biconnected_component_edges(G)   #not implemented for directed type
#print nx.biconnected_component_subgraphs(G)	 #not implemented for directed type
#print nx.articulation_points(G)                #not implemented for directed type'''
##Semiconnectedness
###print nx.is_semiconnected(G)	

'''#Cycles!!!!!
#print nx.cycle_basis(G)     #not implemented for directed type
print nx.simple_cycles(G)   #A generator of lists of cycles  2图一定不能用，因为没有环
print nx.find_cycle(G)'''

#flow_hierarchy
###print nx.flow_hierarchy(G)

#Link Analysis
'''##PageRank!!!!!!
print nx.pagerank(G)        #it will be very big
print nx.pagerank_numpy(G)	#it will be very big
print nx.pagerank_scipy(G)  #it will be very big
print nx.google_matrix(G)   #it will be very big'''
'''##Hits
print nx.hits(G)            #it will be very big
print nx.hits_numpy(G)      #it will be very big
print nx.hits_scipy(G)	    #it will be very big
print nx.hub_matrix(G)      #it will be very big
print nx.authority_matrix(G)#it will be very big'''

'''#Rich Club!!!!!!!
#print nx.rich_club_coefficient(G, normalized=True, Q=100)   #is not defined for directed graphs'''

#print nx.is_chordal(G)      # Directed graphs not supported

###G = G.to_undirected()
###print 'ok,now is undirect'

'''#Clustering!!!!!
print nx.triangles(G)
print nx.clustering(G)                     #it will be very big
print nx.average_clustering(G)             #it will be very big'''

'''#communities
print nx.communicability(G)                          #it will be very big
print nx.communicability_exp(G)                      #it will be very big
print nx.communicability_centrality(G)               #it will be very big
print nx.communicability_centrality_exp(G)           #it will be very big
print nx.communicability_betweenness_centrality(G)   #it will be very big
#print nx.estrada_index(G)                           #array is too big'''

#components
##Connectivity
###print nx.is_connected(G)                     
###print nx.number_connected_components(G)      
'''print nx.connected_components(G)             #you know how to use
print nx.connected_component_subgraphs(G)    #you know how to use
#print nx.node_connected_component(G,n)      #need n so not use'''
##Biconnected components
###print nx.is_biconnected(G)                     
'''print nx.biconnected_components(G)              #you know how to use
print nx.biconnected_component_edges(G)         #you know how to use
print nx.biconnected_component_subgraphs(G)	    #you know how to use
print nx.articulation_points(G)		            #you know how to use'''

'''#Cycles
print nx.cycle_basis(G)                         #it will be very big'''

#Rich Club
#print nx.rich_club_coefficient(G,normalized=False)   #is not defined for directed graphs'''

#print nx.is_chordal(G)      # need too long time


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

'''
#betweenness_centrality
print '$betweenness_centrality'
bc = nx.betweenness_centrality(G)
bclist = []
with open('/home/bill/Desktop/new/betweenness_centrality','w') as f:
	for k,v in bc.items():
		f.write(str(k)+','+str(v)+'\n')	
		bclist.append(v)
f.close()
turntof(retdic(cluster))
plt.savefig("betweenness_centrality.png")

##PageRank!!!!!!
print '$PageRank'
page = []
pagerank = nx.pagerank(G)        #it will be very big
for key,value in pagerank.items():
	page.append(value)
#print retdic(page)
pagedic = retdic(page)
pagedict= sorted(pagedic.iteritems(), key=lambda d:d[0], reverse = True)
PRlist = sorted(pagerank.iteritems(), key=lambda d:d[1], reverse = True)
count = 0
for k,v in PRlist:
	if count < 20:
		print accountinfo.get(k),k
		count += 1
	else:
		break

#drawdic(pagedict)
#plt.savefig("pagerank.png")

#Cycles!!!!!
print '$Cycles'
cyclelist = []
for i in nx.simple_cycles(G):   #A generator of lists of cycles  2图一定不能用，因为没有环
	cyclelist.append(len(i))
#print retdic(cyclelist)
print 'break'
turntof(retdic(cyclelist))
plt.savefig("cycle.png")'''


#G = G.to_undirected()
print 'ok,now is undirect'

'''#Clustering!!!!!
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
richclub =  nx.rich_club_coefficient(G,normalized=False)   #is not defined for directed graphs
rich = []
#for key,value in richclub.items():
#	rich.append(value)
#print richclub
turntof(richclub)
plt.savefig("richclub.png")
'''
