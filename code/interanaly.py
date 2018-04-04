# -*- coding: utf-8 -*-
#! /usr/bin/env python
# test.py

#
#this code aim to extract single edge from internal transactions, single edge means that single call/create to record.
#

import os
import re
import datetime

def deallist(listtemp_s,flagcount,flagcountl):
	typename = listtemp_s[0]        ##use by find
	fromaddr = listtemp_s[1][10:50]  ##addr can use direct
	toaddr = listtemp_s[2][8:48]    ##addr can use direct
	ether_s = listtemp_s[3]
	i = ether_s.find('<Value>')
	j = ether_s.find('</Value>')
	ether = ether_s[7+i:j]          ##str_ether need turn to int_ether or float_ether
	#print fromaddr
	#print toaddr
	#print ether
	flag = 0
	if typename.find('suicide')>=0:
		pass#writewhatiwant(flag)
		flag = 1
	if typename.find(create_s)>0:
		two(fromaddr, toaddr)
		flag = 2
	elif juaddr(toaddr)>0:
		if typename.find('suicide')<0:
			three(fromaddr,toaddr)
			flag = 2
	##print ether
	ethernew = filter(lambda ch: ch in '0123456789.', ether)
	##print ethernew
	if ethernew.find(point_s)>0:
		ether_value = float(ethernew)
	else:
		ether_value = int(ethernew)
	
	if ether_value!=0:
		one(fromaddr, toaddr, ether_value)
		flag = 2
	if flag == 1:
		pass#writewhatiwant(flag)
	'''if typename.find(suicide_s)>0:
		flagcount = flagcount + 1
		if ether_value == 0:
			flagcountl = flagcount+1'''
	if typename.find(suicide_s)>=0:
		pass#writethreesuicide(fromaddr,toaddr)
	if flag == 5:
		#print toaddr
		return 5
	#print "Over!"

'''def juaddr(address):
	with open('/home/bill/Desktop/EtherChain/sc_new.xml','r') as fread:
		for line in fread:
			if line.find(address)>0:
				fread.close()
				return 1
	fread.close()
	return 0'''
def juaddr(address):
	if filelist.get(address,0) == 0:
		return 0
	else: 
		return 1

def writewhatiwant(i):
	if i == 0:
		with open('/home/bill/Desktop/temp/buyaode','a') as f:
			f.write('j\n')
		f.close()
	elif i == 1:
		with open('/home/bill/Desktop/temp/threesuicide','a') as f:
			f.write('g\n')
		f.close()
def gettemp(path_s):
	listtemp = []
	count = 0
	suicidecount = 0
	flagcount = 0
	flagcountl = 0
	with open(path_s,'r') as f:
		print path_s
		for line in f:
			listtemp.append(line)
			if line.find(start)>0:
				listtemp = []
			elif line.find(end)>0:
				if len(listtemp)==6:
					deallist(listtemp,flagcount,flagcountl)
					if listtemp[0].find(suicide_s) >= 0:
						suicidecount += 1				
					count = count +1
					if count == 26196:
						print suicidecount,count,count-suicidecount
						return 0
				
				listtemp = []
	f.close()
	#print count
	#print flagcount
	#print flagcountl

def one(fromaddr, toaddr, ether_value):
	ether_value = str(ether_value)
	with open('/home/bill/Desktop/temp/inter/one','a')     as  f:
		f.write('\n'+fromaddr+','+toaddr+','+ether_value)
	f.close()
def two(fromaddr, toaddr):
	with open('/home/bill/Desktop/temp/inter/two','a')     as  f:
		f.write('\n'+fromaddr+','+toaddr)
	f.close()
def three(fromaddr,toaddr):
	with open('/home/bill/Desktop/temp/inter/three','a')     as  f:
		f.write('\n'+fromaddr+','+toaddr)
	f.close()


with open('/home/bill/Desktop/temp/inter/one','a')     as  f:
	f.write('Source,Target,Weight')
f.close()
with open('/home/bill/Desktop/temp/inter/two','a')     as  f:
	f.write('Source,Target')
f.close()
with open('/home/bill/Desktop/temp/inter/three','a')   as  f:
	f.write('Source,Target')
f.close()


start = '<Internal>\n'
end = '</Internal>\n'
call_s = 'call'
create_s = 'create'
suicide_s = 'suicide'
point_s = '.'
filelist = {}
with open('/home/bill/Desktop/EtherChain/sc_new.xml','r') as fre:	
	for line in fre:
		line = line[12:52]
		'''if filelist.get(line,0) != 0:
			#print line
			count = count + 1''' 
		filelist.update({line:1})
	#print count
fre.close()
c = []
path = "/home/bill/Desktop/selectdraw/inter"
starttime = datetime.datetime.now()
files = os.listdir(path)
for file in files:
	gettemp(path+"/"+file)
endtime = datetime.datetime.now()
print (endtime - starttime).seconds
print len(filelist)
filelist.clear()
