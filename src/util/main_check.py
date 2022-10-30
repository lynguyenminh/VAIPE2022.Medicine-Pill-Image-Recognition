from difflib import SequenceMatcher
from util.clean_string import clean_drugname
# from clean_string import clean_drugname


# score of 2 strings
def similar(a, b):
    a = clean_drugname(a)
    b = clean_drugname(b)
    return SequenceMatcher(None, a, b).ratio()

# kiem tra 1 ten thuoc co thuoc danh sach ten thuoc trong prescription khong
def check_pill_in_pres(pill_name, name_in_prescription, score):
    '''
    This function check pill in prescription by calculate similar drugname.
    '''
    for i in name_in_prescription: 
        if similar(pill_name, i) > score: 
            return True
    return False

# kiem tra xem danh sach ten thuoc co thuoc prescription ko
def decision_in_out(name_in_pill, name_in_prescription, score):
    count = [check_pill_in_pres(i, name_in_prescription, score) for i in name_in_pill]
    return sum(count)

# print(decision_in_out(['amoxicilin500mg500mg', 'fabamox500500mg', 'novoxim50005g'], ['novoxim50005gtru', 'kavasdin55mg', 'mypara500500mg'], 0.80))
