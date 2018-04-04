# -*- coding: utf-8 -*-
#! /usr/bin/env python
# test.py
import os
import re
import datetime

def deallist(listtemp_s):
	typename = listtemp_s[0]        ##use by find
	fromlist = listtemp_s[3].split(' ')
	tolist = listtemp_s[4].split(' ')
	etherlist = listtemp_s[5].split(' ')
	fromaddr = fromlist[1]
	if tolist[1].find(contract_s)>=0:
		toaddr = tolist[2][0:42]
	else:
		toaddr = tolist[1][0:42]
	if etherlist[2].find(wei_s)>=0:
		ether = zero
	else:
		ether = etherlist[1]
	if typename.find(sc_s)>0:
		if listtemp_s[4].find(created_s)>0:
			two(fromaddr, toaddr)
			#print fromaddr
			#print toaddr
			#print ether
		else:
			three(fromaddr,toaddr)
	#print fromaddr
	#print toaddr
	#print ether
	ethernew = filter(lambda ch: ch in '0123456789.', ether)
	##print ethernew
	if ethernew.find(point_s)>0:
		ether_value = float(ethernew)
	else:
		ether_value = int(ethernew)

	if ether_value!=0:
		one(fromaddr,toaddr,ether_value)
	#print "Over!"

def juaddr(address):
	with open('/home/bill/Desktop/EtherChain/sc_new.xml','r') as fread:
		for line in fread:
			if line.find(address)>0:
				fread.close()
				return 1
	fread.close()
	return 0

def gettemp(path_s,count):
	listtemp = []
	count = 0
	suicidecount = 0
	with open(path_s,'r') as f:
		for line in f:
			listtemp.append(line)
			if line.find(start)>0:
				listtemp = []
			elif line.find(end)>0:
				if len(listtemp)==6:
					if count == 23806:
						return 0
					count = count + 1
					deallist(listtemp)
				listtemp = []
	print count
	f.close()
	return count

def one(fromaddr, toaddr, ether_value):
	ether_value = str(ether_value)
	with open('/home/bill/Desktop/temp/trans/one','a')     as  f:
		f.write('\n'+fromaddr+','+toaddr+','+ether_value)
	f.close()
def two(fromaddr, toaddr):
	with open('/home/bill/Desktop/temp/trans/two','a')     as  f:
		f.write('\n'+fromaddr+','+toaddr)
	f.close()
def three(fromaddr,toaddr):
	with open('/home/bill/Desktop/temp/trans/three','a')   as  f:
		f.write('\n'+fromaddr+','+toaddr)
	f.close()


with open('/home/bill/Desktop/temp/trans/one','a')     as  f:
	f.write('Source,Target,Weight')
f.close()
with open('/home/bill/Desktop/temp/trans/two','a')     as  f:
	f.write('Source,Target')
f.close()
with open('/home/bill/Desktop/temp/trans/three','a')   as  f:
	f.write('Source,Target')
f.close()


count = 0
start = '<TxHash>'
end = '</Value>'
wei_s = 'wei'
zero = '0'
client_s = 'Client'
sc_s = 'smart contract'
contract_s = 'Contract'
suiside_s = 'suicide'
created_s = 'Created'
point_s = '.'
path = "/home/bill/Desktop/selectdraw/trans"
starttime = datetime.datetime.now()
files = os.listdir(path)
for file in files:
	count = gettemp(path+"/"+file,count)
endtime = datetime.datetime.now()
print (endtime - starttime).seconds
