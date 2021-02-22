import string,math,sys

def read_files(file1,file2):
	f1=open(file1,'r')
	f2=open(file2,'r')
	text1=f1.read()
	text2=f2.read()
	return text1,text2

def get_word_frequency(text1,text2):
	freq1={}
	freq2={}
	translation_table=str.maketrans(string.punctuation+string.ascii_uppercase," "*len(string.punctuation)+string.ascii_lowercase)
	word_list_1=text1.translate(translation_table).split()
	word_list_2=text2.translate(translation_table).split()
	for word in word_list_1:
		if word in freq1:
			freq1[word]+=1
		else:
			freq1[word]=1
	for word in word_list_2:
		if word in freq2:
			freq2[word]+=1
		else:
			freq2[word]=1
	return freq1,freq2

def dot_product(A,B):
	dot=0
	for i in A:
		dot+=A[i]*B.get(i,0)
	return dot

def cosine_distance(A,B):
	cos_theta=dot_product(A,B)/math.sqrt(dot_product(A,A)*dot_product(B,B))
	return 1-cos_theta

if __name__ == "__main__":

	n=len(sys.argv)
	if(n!=3):
		print(f"correct usage: python3 {sys.argv[0]} text_file_1 text_file_2")
		sys.exit()
	text1,text2=read_files(sys.argv[1],sys.argv[2])
	freq1,freq2=get_word_frequency(text1,text2)
	distance=cosine_distance(freq1,freq2)
	print(f"cosine distance between {sys.argv[1]} and {sys.argv[2]} = {distance}")