import requests

API_URL = "https://api-inference.huggingface.co/models/codellama/CodeLlama-7b-hf"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": "def sort(d):#sort a dict by value ascendingly and return a dict",
})
#addcomment
print('myOutputis ', output)

def sortOne(d):
    return {k:v for k, v in sorted(d.items(), key = lambda item: item[1])} 

def sortTwo(d): 
      return sorted(d.items(),key=lambda x:x[1]) 

def sortThree(d):
    print(sorted(d.items(),key=lambda x:(x[1],x[0])))

def reverse_sort(d):#sort a dict by value decendingly and return a dict object
    print(sorted(d.items(),key=lambda x:(x[1],x[0]),reverse=True))    

thisValue=sortOne({"model": "tara","brand": "sam","name": "jack"})

print("This value is: ", thisValue)


thisValue2 = sortTwo({"model": "tara","brand": "sam","name": "jack"})

print(thisValue2)

sortThree({"model": "tara","brand": "sam","name": "jack"})
reverse_sort({"model": "tara","brand": "sam","name": "jack"})