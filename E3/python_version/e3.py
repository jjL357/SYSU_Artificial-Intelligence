# coding=gb2312
import os
import copy
import string
class Predicate:#��ν�ʺ͸���(����)
    def __new__(cls):
        return super().__new__(cls)
    def __init__(self):
        self.pred = ''#ν��
        self.indiv = []#ν�ʵĸ�������б�
        self.isfunc = []#�ж��Ƿ��Ǻ���
        self.flag= True #�ж����ֵ
    def printPredicate(self):#���ش�ӡν��
        tmp = ''
        if(self.flag == False):
            tmp += '~'#�Ƿ�����~��ʾ
        tmp += self.pred
        tmp += '('
        for i in range(len(self.indiv)):
            tmp += self.indiv[i]
            if(not (i == len(self.indiv)-1) ): tmp += ','
        tmp += ')'
        return tmp

class Clause:#���Ӿ�
    def __new__(cls):
        return super().__new__(cls)
    def __init__(self) :
        self.index = 0#��¼���Ӿ伯���±�
        self.pre = []#Predicate���б�
        self.son = []#���������������������������
        self.sonson = []#��¼1a��2b�е���ĸ���ɹ���Ӿ����һ���ֹ��
        self.msg = ''#��¼��һ
        self.printindex = 0#�����������
    def printClause(self):#��ӡ����Ӿ�
        tmp = ''
        if(len(self.pre)>1):tmp += '('
        for i in range(len(self.pre)):
            tmp += self.pre[i].printPredicate()
            if(not(i == len(self.pre)-1 )):tmp += ','
        if(len(self.pre)>1):tmp += ')'
        return tmp

def ifvariable(s):#�ж��Ƿ�Ϊ�����ļ򵥺���
    flag = False
    variables = ['w','v','u','x','y','z','s']#ʾ���еı���
    return s in variables
        
def printlist(lis):#��ӡ����Ӿ伯
    for i in range(len(list)):
        print(i,':',list[i].printClause())
        for j in range(len(list[i].pre)):
            print(lis[i].pre[j].pred,lis[i].pre[j].indiv,lis[i].pre[j].flag)
        print('--------------------------------')   

def pri(s1,s2):#�ж���ô������
    if(not(len(s1) == len(s2) )):
        return len(s1)<len(s2)
    return s1<s2

def load(ans,index,empty_son):#�������̽�������Ӿ������Ӿ�
    
    if(len(ans[index].son) == 0):
        return
    else :
        empty_son.append(copy.deepcopy(ans[index]))
        load(ans,ans[index].son[0],empty_son)
        load(ans,ans[index].son[1],empty_son)

def transform(ans ,f):#��������룬������Ϣ�����Ӧ������
    n = int(f.readline())#һ��һ�ж����ļ�
    print('����:')
    print(n)
    for j in range(n): 
        temp = f.readline()#һ��һ�ж����ļ�
        print(temp,end='')
        q= [] #��ν��
        e= [] #�Ÿ���
        temp = temp.strip()#�Ѷ���Ļ��з�ɾ��
        if temp[0] == '(':#ɾ�������������
            temp = temp[1:len(temp)-1:]
        
        temp2 = []
        cur = 0
        for i in range(len(temp)-1):#һ���־�������ж��ν�ʣ���ν�ʷֿ�����temp2
            if temp[i] == ')' and temp[i+1] == ',' :
                temp2.append(temp[cur:i+1:])
                cur = i+2
        temp2.append(temp[cur::])

        for a in temp2:#��ÿ��ν�ʵ�ν�ʺ͸������p,e
            temp3 = 0
            for i in range(0,len(a)):
                if a[i] == '(':
                    temp3 = i
                    break
            q.append(a[0:temp3:])

            temp4 = a[temp3:len(a):]
            if temp4[0] == '(':temp4=temp4[1:len(temp4)-1:]#ɾ�����������
            temp4 = temp4.split(',')

            for i in range(0,len(temp4)):
                if(temp4[i][0] == ' '):temp4[i] = temp4[i][1::]#ɾ���ո�
            e.append(temp4)

        ans.append(Clause())#ans�����Ӿ�  
        ans[-1].index = len(ans)-1 

        for i in range(0,len(q)):#��ν�ʺ͸������ans���Ӿ���
            ans[-1].pre.append(Predicate())
            if(q[i][0] == '~'):
                q[i] = q[i][1::]
                ans[-1].pre[-1].flag = False
            elif(len(q[i]) > 1 and q[i][1] == '~'):
                q[i] = q[i][2::]
                ans[-1].pre[-1].flag = False
            elif(q[i][0] == ' '):
                q[i] = q[i][1::]
            ans[-1].pre[-1].pred = copy.deepcopy(q[i])
            ans[-1].pre[-1].indiv = copy.deepcopy(e[len(ans[-1].pre)-1])

        for i1 in range(len(ans[-1].pre)):#�жϺ��������Ȳ�д
            for i2 in range (len(ans[-1].pre[i1].indiv)):
                    for i3 in range(len(ans[-1].pre[i1].indiv[i2])):
                        if('(' in ans[-1].pre[i1].indiv[i2][i3]):ans[-1].pre[i1].isfunc.append(i3)

def merge(ans,cur,index):
    j = 0
    size = len(ans)
    endflag = False#�ж��Ƿ���ֿ��Ӿ�
    while j < len(ans) and not endflag:#���
        for k in range(len(ans[j].pre)):
            s = ans[j].pre[k].pred
            f = ans[j].pre[k].flag
            size = len(ans)
            i = j+1
            while i < size and not endflag:
                for l in range(len(ans[i].pre)):
                    if(j in ans[i].son):continue
                    s1 = ans[i].pre[l].pred
                    f1 = ans[i].pre[l].flag
                    if(s == s1 and not(f == f1 )): 
                        flag = False
                        dic1 = {}
                        dic2 = {}
                        for o in range(len(ans[i].pre[l].indiv)):
                            if((ifvariable(ans[i].pre[l].indiv[o]) and not ifvariable(ans[j].pre[k].indiv[o])) or (ifvariable(ans[j].pre[k].indiv[o]) and not ifvariable(ans[i].pre[l].indiv[o]))or ans[i].pre[l].indiv[o] == ans[j].pre[k].indiv[o] ):                    
                                if(pri(ans[i].pre[l].indiv[o] ,ans[j].pre[k].indiv[o])):
                                    flag = True
                                    dic1[ans[i].pre[l].indiv[o]] = ans[j].pre[k].indiv[o]
                                    dic2[ans[j].pre[k].indiv[o]] = ans[i].pre[l].indiv[o]
                                else:
                                    flag=True
                                    dic1[ans[j].pre[k].indiv[o]] = ans[i].pre[l].indiv[o]
                                    dic2[ans[i].pre[l].indiv[o]] = ans[j].pre[k].indiv[o]
                            else :
                                flag = False
                                break
            
                        if(not flag):
                            continue
                        
                        ans.append(Clause())

                        for p in range(len(ans[j].pre)):
                            if(p == k):continue
                            temp = copy.deepcopy(ans[j].pre[p])
                            for e in range(len(temp.indiv)):
                                if(temp.indiv[e] in dic1):
                                    temp.indiv[e] = dic1[temp.indiv[e]]
                            ans[-1].pre.append(temp)

                        for p in range(len(ans[i].pre)):
                            if(p == l):continue
                            temp = copy.deepcopy(ans[i].pre[p])
                            for e in range(len(temp.indiv)):
                                if(temp.indiv[e] in dic1):
                                    temp.indiv[e] = dic1[temp.indiv[e]]
                            ans[-1].pre.append(copy.deepcopy(temp))
                                                
                        for item in dic1.values():
                            if(not(dic2[item] == item)):
                                ans[-1].msg += dic2[item]+ "=" + item + " "
                        p = 0
                        while p < len(ans[-1].pre):
                            p1 = p+1
                            while p1<len(ans[-1].pre):
                                if(ans[-1].pre[p].pred==ans[-1].pre[p1].pred and ans[-1].pre[p].indiv == ans[-1].pre[p1].indiv):
                                    del ans[-1].pre[p1]
                                    p1 -= 1
                                p1 += 1
                            p += 1
                        ans[-1].son.append(j)
                        ans[-1].son.append(i)
                        ans[-1].sonson.append(k)
                        ans[-1].sonson.append(l)
                        ans[-1].index=len(ans)-1
                        if(ans[-1].printClause()==''):
                            endflag = True
                            index = len(ans)-1
                i = i+1
        j = j+1
    return index

def printans(cur,ans,index,alpha):
    cur1 = cur
    b= []
    load(ans,index,b)
    b.sort(key=lambda x:(x.son[0],x.son[1]))#���������Ӿ���Ӿ���Ⱥ�˳������
    for i in range(len(b)):
        print('R[',end='')
        if(b[i].son[0] >= cur):
            flag=False
            for j in range(i):
                if(ans[b[i].son[0]].index == b[j].index):
                    print(b[j].printindex,end='')
                    flag = True
                    break
        else :
            print(b[i].son[0]+1,end='')
        print(alpha[b[i].sonson[0]],end='')
        print(',',end='')

        if(b[i].son[1] >= cur):
            flag=False
            for j in range(i):
                if(ans[b[i].son[1]].index == b[j].index):
                    print(b[j].printindex,end='')
                    flag=True
                    break
        else :
            print(b[i].son[1]+1,end='')

        print(alpha[b[i].sonson[1]],end='')
        print(']',end='')
        print('{',end='')
        print(b[i].msg,end='')
        print('}=',end='')
        print(b[i].printClause())
        cur1 += 1
        b[i].printindex = cur1

#������
def main():
    alpha = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']  
    f = open(r"ai\E3\E3_code\input.txt",'r')
    ans = []#����Ӿ伯
    transform(ans,f)#�����������ת��
    cur = len(ans)#��¼��ʼ�־����
    index = -1#��¼���Ӿ����±�
    index = merge(ans,cur,index)#���
    if index == -1:
        print("Unsuccessful")#�޷��������Ӿ�
    else:
        print('���:')
        printans(cur,ans,index,alpha)#�����

if __name__ == '__main__':
    main()