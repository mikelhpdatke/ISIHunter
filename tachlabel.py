import sys
import  json
if __name__=="__main__":
	print(sys.argv)
	fi = sys.argv[1]
	with open(fi) as f:
		content = f.readlines()
	content =[x.strip() for x in content]

	dict = {}
    
	print("Loading table..!")
	ans = json.load(open("table_final.code"))

	f_label = open('label.out', "w+") 
	f_nolabel = open('nolabel.out', 'w+')
	for x in content:
		list = str(x).split(',')
		list[1] = ans[list[1]]
		list[2] = ans[list[2]]
		list[3] = ans[list[3]]
		list[len(list) - 1] = ans[list[len(list) - 1]]
		s = ""
		for i in list:
			s = s + str(i) + ','
		string_nolabel = list[:-1]
		s_nolabel = ""
		for i in string_nolabel:
			s_nolabel = s_nolabel + str(i) + ','
            	f_label.write(s[:len(s) - 1] + "\n")
            	f_nolabel.write(s_nolabel[:len(s_nolabel) - 1] + "\n")
    	f_label.close()
    	f_nolabel.close()        
