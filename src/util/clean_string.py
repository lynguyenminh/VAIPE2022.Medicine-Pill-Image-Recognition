import re, string

def clean_drugname(drug_name):
    drug_name = drug_name.lower() #lowercase drug_name
    drug_name=drug_name.strip()  #get rid of leading/trailing whitespace 
    drug_name = re.compile('[%s]' % re.escape(string.punctuation)).sub('', drug_name)  #Replace punctuation with space. Careful since punctuation can sometime be useful
    
    drug_name=re.sub(r'[^\w\s]', '', str(drug_name).lower().strip())
    drug_name = re.sub(r'\s+',' ',drug_name) #\s matches any whitespace, \s+ matches multiple whitespace, \S matches non-whitespace 
    
    # replace 0.5g -> 500mg
    for i in range(1, 9):
        drug_name = drug_name.replace(''.join(['0', str(i), 'g']), ''.join([str(i), '00mg']))
    
    # replace 0.45g -> 450mg
    for i in range(10, 99):
        drug_name = drug_name.replace('0' + str(i) + 'g', str(i * 10) + 'mg')
    
    # replace o.5g -> 500mg
    for i in range(1, 9):
        drug_name = drug_name.replace('o' + str(i) + 'g', str(i) + '00mg')
    
    # replace 500 500mg -> 500mg
    for i in ['500mg', '1000mg', '500', '1000', '250', '125', '850', '800', '250mg', '850mg', '800mg', '300mg', '300', '20mg', '10mg', '200mg', '30mg', '30', '60mg', '125mg', '25mg', '145mg', '16mg', '4mg', '175mg', '175', '16', '4', '145', '25', '60', '200', '20', '10']:
        if drug_name.count(i) > 1: 
            drug_name = drug_name.replace(i, '', 1)

    drug_name = re.sub('\s+', ' ', drug_name)  #Remove extra space and tabs
    drug_name=re.sub(r'[^\w\s]', '', str(drug_name).lower().strip())
    drug_name = re.sub(r'\s+',' ',drug_name) #\s matches any whitespace, \s+ matches multiple whitespace, \S matches non-whitespace 

    return drug_name

def post_process_text(text):
    if text == '':
        return ''

    # chuyen thanh chu viet thuong
    text = text.lower()

    # phat hien nhan dien sai
    for i in ['chẩn đoán', 'bệnh', 'chứng', 'sl:', 'lời dặn', 'bố', 'mẹ', 'sáng', 'trưa', 'tối', 'đoán', 'tái khám', 'khám lại', 'xơ vữa động mạch', 'uống trước khi ăn', 'uống sau khi ăn', 'viêm họng cấp', 'tìnhtrạng', 'tình trạng']:
        if i in text:
            return ''
    # xoa ngoac kep (dau hoac cuoi)
    text = text[1:] if text[0] == '"' else text
    text = text[:-1] if text[-1] == '"' else text

    # xoa so thu tu
    # ex: '1) GLUCOFAST 500 500mg'
    temp = text.split(' ')
    if temp[0][-1] == ')':
        text= text.replace(temp[0], '')

    # xoa cach va cac ki tu dat biet
    for i in [',', '.', '+', '-', '%', ' ', '\'', '(', ')']:
        text = text.replace(i,'')

    text = text.replace('dường', 'dưỡng')

    for i in text: 
        if i.isnumeric() or i == ')': 
            text = text[1:]
        else: 
            break

    return text