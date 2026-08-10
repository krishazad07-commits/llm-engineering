def greet(name:str)-> str:
    return f"hello,{name}"

def add(a:int,b:int)->int:
    return a+b 

def find_user(id:str)->str| None:
    if id=="42":
        return"krish"
    return"none"

scores:list[int]=[30,40,50,60]
user: dict[str,str|int]={"name":"krish","age":22}

print(greet("krish"))
print(add(30,50))
print(find_user("42"))
print(scores)
print(user)
