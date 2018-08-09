import csv

def init(str):
    param = ''
    while(True):
        if (str[-1] == '*' or str[-1] == ' '):
            break
        param = str[-1] + param
        str = str[:-1]
    return [str, param]

with open('syscall.txt',encoding='utf-8') as f:
    with open('hook_fuc.h', 'w+', encoding='utf-8') as fo1:
        with open('hook_sys.h', 'w+',encoding='utf-8') as fo2:
            content = csv.reader(f, delimiter=',')
            for x in content:
                sys_name = x[0]
                res = 'asmlinkage long (*real_sys_' + sys_name + ')'
                res_more = 'asmlinkage long _hook_sys_' + sys_name
                res = res + '('
                res_more = res_more + '('
                params = ''
                for i in range(1, 7):
                    if (x[i] != ''):
                        after_exe = init(x[i])
                        res = res + after_exe[0] + ','
                        res_more = res_more + x[i] + ','
                        params = params + after_exe[1] + ','
                if (res[-1] == ','):
                    res = res[:-1]
                if (res_more[-1] == ','):
                    res_more = res_more[:-1]
                    params = params[:-1]
                res = res + ');'
                res_more = res_more + \
                ')\n{\n\tprintk(KERN_ERR "syscall:' +\
                sys_name +\
                '\\n");\n\treturn real_sys_' +\
                sys_name +\
                '(' +\
                params +\
                ');'+\
                '\n}\n'
                fo1.write(res+'\n')
                fo2.write(res_more + '\n')
