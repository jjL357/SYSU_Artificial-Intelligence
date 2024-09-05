# coding=gb2312
class Predicate:  # �����ȶ���һ��ν���������Ӿ��е�ν�ʽ��д洢
    element = []

    def __init__(self, str_in):
        self.element = []
        if len(str_in) != 0:
            if str_in[0] == ',':  # ��ԭ�����ڷָ�ν�ʵ� , ȥ��
                str_in = str_in[1:]
            tmp = ""
            for i in range(len(str_in)):
                tmp += str_in[i]
                if str_in[i] == '(' or str_in[i] == ',' or str_in[i] == ')':
                    self.element.append(tmp[0:-1])
                    tmp = ""

    def new(self, list_in):
        for i in range(len(list_in)):
            self.element.append(list_in[i])

    def rename(self, old_name, new_name):
        for i in range(len(old_name)):
            j = 1
            while j < len(self.element):
                if self.element[j] == old_name[i]:
                    self.element[j] = new_name[i]
                j = j + 1

    def get_pre(self):  # ����ν�ʵ�ǰ׺�Ƿ�Ϊ"?"
        return self.element[0][0] == "~"

    def get_name(self):  # ����ν������
        if self.get_pre():
            return self.element[0][1:]
        else:
            return self.element[0]


def print_clause(clause_in):  # ��ν���б��ӡ��ԭ�����Ӿ�
    tmp = ""
    if len(clause_in) > 1:
        tmp = tmp + "("
    for i in range(len(clause_in)):
        tmp = tmp + clause_in[i].element[0] + "("
        for j in range(1, len(clause_in[i].element)):
            tmp = tmp + clause_in[i].element[j]
            if j < len(clause_in[i].element) - 1:
                tmp = tmp + ","
        tmp = tmp + ")"
        if i < len(clause_in) - 1:
            tmp = tmp + ","
    if len(clause_in) > 1:
        tmp = tmp + ")"
    if (tmp != ""):
        print(tmp)


def print_msg(key, i, j, old_name, new_name, set_of_clause):
    tmp = str(len(set_of_clause)) + ": R[" + str(i + 1)  # ��������Ϣ�� R[A1= 1 ,A2= 6 a ](x = tony,)
    if len(new_name) == 0 and len(set_of_clause[i]) != 1:
        tmp = tmp + chr(key + 97)
    tmp = tmp + ", " + str(j + 1) + chr(key + 97) + "]("
    for k in range(len(old_name)):
        tmp = tmp + old_name[k] + "=" + new_name[k]
        if k < len(old_name) - 1:
            tmp = tmp + ", "
    tmp = tmp + ") = "
    print(tmp, end="")


def end_or_not(new_clause, set_of_clause):
    if len(new_clause) == 0:  # �����ɵ�new_clause�Ѿ�Ϊ��
        print("[]")
        return True
    if len(new_clause) == 1:  # �������е��Ӿ����Ƿ���������Ӿ以��
        for i in range(len(set_of_clause) - 1):  # set_of_clause[j]����һ��ν�ʵ�ȡ����Ӿ�
            if len(set_of_clause[i]) == 1 and new_clause[0].get_name() == set_of_clause[i][0].get_name() and new_clause[0].element[1:] == \
                    set_of_clause[i][0].element[1:] and new_clause[0].get_pre() != set_of_clause[i][0].get_pre():
                print(len(set_of_clause) + 1, ": R[", i + 1, ", ", len(set_of_clause), "]() = []", sep="")
                return True
    return False  # ����������������


def main():
    set_of_clause = []
    print("���ȣ��������Ӿ�������")
    num_of_clause = input()
    print("���棬������", num_of_clause, "���Ӿ䣺")
    for i in range(int(num_of_clause)):
        clause_in = input()
        if clause_in == "":
            print("���������������Ϊ�գ��������˳�")
            return
        if clause_in[0] == '(':  # ����Ӿ�����������ţ���ȥ��
            clause_in = clause_in[1:-1]
        clause_in = clause_in.replace(' ', '')  # ����Ӿ����пո���ȥ��
        set_of_clause.append([])  # һ���б���������Ӿ��ִ洢
        tmp = ""  # ���ڲ���Ӿ�ʹ�õ��м����
        for j in range(len(clause_in)):  # ��ִ洢���б���
            tmp += clause_in[j]
            if clause_in[j] == ')':  # ��')'��Ϊ��β�ָ�ɶ��ν�ʹ�ʽ
                if j + 1 != num_of_clause:
                    clause_tmp = Predicate(tmp)  # ����һ��ν�ʹ�ʽ��Predicate�ı���
                    set_of_clause[i].append(clause_tmp)  # ���뵽�Ӿ伯�ĵ�i���Ӿ���
                tmp = ""

    for i in range(len(set_of_clause)):  # ������ո�������Ӿ伯
        print_clause(set_of_clause[i])

    status = True
    while status:
        for i in range(len(set_of_clause)):
            if not status:
                break
            if len(set_of_clause[i]) == 1:  # ֻ��һ��ν�ʵ��Ӿ�set_of_clause[i]�������洦��
                for j in range(0, len(set_of_clause)):  # ���������Ӿ���бȽ�
                    if not status:
                        break
                    if i == j:  # �����Լ��Ƚ�
                        continue
                    old_name = []
                    new_name = []  # �����ɱ���ת��ΪԼ������
                    key = -1  # -1��ʾ���Ӿ��ͬ��ν�ʲ��ܽ�����ȥ
                    for k in range(len(set_of_clause[j])):  # ���Ӿ�set_of_clause[j]������ͬ��ν�ʣ��ҿ�����ȥ������keyΪ��λ��
                        if set_of_clause[i][0].get_name() == set_of_clause[j][k].get_name() and set_of_clause[i][
                            0].get_pre() != set_of_clause[j][k].get_pre():
                            key = k
                            for l in range(len(set_of_clause[j][k].element) - 1):  # �ҵ����Ի����ı�������¼
                                if len(set_of_clause[j][k].element[l + 1]) == 1:  # �����ɱ���
                                    old_name.append(set_of_clause[j][k].element[l + 1])
                                    new_name.append(set_of_clause[i][0].element[l + 1])
                                elif len(set_of_clause[i][0].element[l + 1]) == 1:
                                    old_name.append(set_of_clause[i][k].element[l + 1])
                                    new_name.append(set_of_clause[j][0].element[l + 1])
                                elif set_of_clause[j][k].element[l + 1] != set_of_clause[i][0].element[l + 1]:
                                    key = -1
                                    break
                            break
                    if key == -1:  # ������ ��ȥ �������Ӿ�
                        continue
                    new_clause = []  # ��¼���ɵ����Ӿ�
                    for k in range(len(set_of_clause[j])):
                        if k != key:  # λ��Ϊkey���Ѿ�����ȥ�ˣ����Բ������Ӿ���
                            p = Predicate("")
                            p.new(set_of_clause[j][k].element)
                            p.rename(old_name, new_name)
                            new_clause.append(p)
                    if len(new_clause) == 1:  # �ж��Ƿ����ɵ��Ӿ��Ƿ��������ظ������ж��Ƿ��������Ӿ䣩
                        for k in range(len(set_of_clause)):
                            if len(set_of_clause[k]) == 1 and new_clause[0].element == set_of_clause[k][0].element:
                                key = -1
                                break
                    if key == -1:  # ������ɵ��Ӿ��Ѵ��ڣ����������Ӿ伯�Ĺ���
                        continue
                    set_of_clause.append(new_clause)  # ���ɵ��µ��Ӿ������Ӿ伯��
                    print_msg(key, i, j, old_name, new_name, set_of_clause)  # ����������Ӿ�������Ϣ
                    print_clause(new_clause)  # ��������Ӿ�
                    if end_or_not(new_clause, set_of_clause):  # �ж��Ƿ�Ӧ�ý���������
                        status = False
                        break
            #  ����Ĳ��֣���Ϊ�˽����������֮���һЩ���� ��Ϊ����ĳ����õ��Ĺ����� (A)and(?A,B,C,...) => (B,C,...)
            else:  # set_of_clause[i]���ж��ν�ʵ��Ӿ�
                for j in range(0,
                               len(set_of_clause)):  # �ҿ�ʹ�ù��� (?A,B,C,...)and(A,B,C,...) => (B,C,...) ���Ӿ�set_of_clause(j)
                    key = -1
                    if i != j and len(set_of_clause[i]) == len(set_of_clause[j]):
                        for k in range(len(set_of_clause[i])):
                            if set_of_clause[i][k].element == set_of_clause[j][k].element:
                                # ʵ������У�Ӧ�ý�һ�����Ǹ��ֿɽ��б������������
                                continue
                            elif set_of_clause[i][k].get_name() == set_of_clause[j][k].get_name() and set_of_clause[i][
                                                                                                          k].element[
                                                                                                      1:] == \
                                    set_of_clause[j][k].element[1:]:
                                # ��Ҫ�������жϱ������������
                                if key != -1:  # �����Ѿ�����һ�����ȵ�������޷�ʹ�øù����������
                                    key = -1
                                    break
                                key = k
                            else:
                                key = -1
                                break
                    if key == -1:
                        continue
                    new_clause = []
                    for k in range(len(set_of_clause[i])):
                        if k != key:
                            p = Predicate("")
                            p.new(set_of_clause[j][k].element)
                            new_clause.append(p)
                    if len(new_clause) == 1:  # �ж��Ƿ����ɵ��Ӿ��Ƿ��������ظ������ж��Ƿ��������Ӿ䣩
                        for k in range(len(set_of_clause)):
                            if len(set_of_clause[k]) == 1 and new_clause[0].element == set_of_clause[k][0].element:
                                key = -1
                                break
                    if key == -1:  # ������ɵ��Ӿ��Ѵ��ڣ����������Ӿ伯�Ĺ���
                        continue
                    set_of_clause.append(new_clause)
                    print_msg(key, i, j, [], [], set_of_clause)  # ����������Ӿ�������Ϣ
                    print_clause(new_clause)  # ��������Ӿ�
                    if end_or_not(new_clause, set_of_clause):  # �ж��Ƿ�Ӧ�ý���������
                        status = False
                        break
    print("Success!")


if __name__ == '__main__':
    main()