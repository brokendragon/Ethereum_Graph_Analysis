7c20-wcc is the wcc(weakly-connect-component) that obtains address 7c20 at the CCG.
3898-wcc is the wcc(weakly-connect-component) that obtains address 3898 at the CCG.
We can use these files with CCG to get each address's outdegree(smart contracts they has created).
We can use these files with CIG to get each address's indegree(times they has been called).
We can use these files with MFG to get each address's sum of edges weight(ether they has transfered).
And use Algorithm 1 in paper, we can do the application A(Attack Forensics).

Algorithm 1 Detection of abnormal contract creation
Inputs: x, the detected account
MFG, money flow graph
CCG/CIG, contract creation/invocation graphs
T1,T2,T3, thresholds
Outputs: True/False, x is abnormal/benign
1 sc_set = created_sc(CCG, x);
2 if size(sc_set) < T1 return False;
3 for each node y in sc_set
4 caller_set = inedge(CIG, y);
5 for each edge z in caller_set
6 num += z.weight;
7 sender_set = inedge(MFG, y);
8 for each edge s in sender_set
9 value += s.weight;
10 if num > T2 × size(sc_set) || value > T3 × size(sc_set)
11 return False;
12 else return True;
