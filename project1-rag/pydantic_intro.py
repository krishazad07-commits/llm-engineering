from pydantic import BaseModel, ValidationError

class User(BaseModel):
    name:str
    age:int
    email:str| None=None

user1= User(name="Krish",age=22,email=None)
print(user1)
print(f"name:{user1.name}")
print(f"age:{user1.age}")
print(f"email:{user1.email}")

# Convert model to a dict
print("\n--- As dict ---")
print(user1.model_dump())

# Convert model to a JSON string
print("\n--- As JSON ---")
print(user1.model_dump_json())

# Pretty JSON with indentation
print("\n--- Pretty JSON ---")
print(user1.model_dump_json(indent=2))

try:
    # This will raise a ValidationError because age should be an int
    bad_user = User(name="BadUser", age="not_an_int", email="bad@example.com")
    print(bad_user)
except ValidationError as e:
    print("ValidationError caught:")
    print(e)

# Show the errors as structured data
print("\n--- Errors as structured data ---")
try:
    bad_user = User(name="Rahul", age="not a number")
except ValidationError as e:
    for error in e.errors():
        print(error)

# Simulate an LLM returning JSON
print("\n--- Parsing LLM output (good) ---")
llm_response = '{"name": "Meera", "age": 30, "email": "meera@example.com"}'
parsed = User.model_validate_json(llm_response)
print(parsed)
print(f"Meera's email is: {parsed.email}")


# Simulate an LLM returning slightly wrong JSON
print("\n--- Parsing LLM output (bad) ---")
bad_response = '{"name": "Vikram", "age": "thirty"}'
try:
    parsed_bad = User.model_validate_json(bad_response)
except ValidationError as e:
    print("LLM gave us garbage:")
    print(e)


# Simulate an LLM missing a required field
print("\n--- Parsing LLM output (missing field) ---")
missing = '{"name": "Priya"}'  # no age
try:
    parsed_missing = User.model_validate_json(missing)
except ValidationError as e:
    print("LLM forgot a field:")
    print(e)

print("\n--- Experiment 1: negative age ---")
weird_user = User(name="Test", age=-5)
print(weird_user)

print("\n--- Experiment 2: age as string containing digits ---")
coerced_user = User(name="Test", age="22")
print(coerced_user)
print(f"Type of age: {type(coerced_user.age)}")

print("\n--- Experiment 3: extra field ---")
extra_user = User(name="Test", age=22, city="Ahmedabad")
print(extra_user)