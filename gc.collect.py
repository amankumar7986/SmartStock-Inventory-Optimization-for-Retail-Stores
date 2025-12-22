import gc

class Demo:
    def __init__(self, name):
        self.name = name
        self.ref = None

    def __del__(self):
        print(f"Object {self.name} is garbage collected")

# Enable garbage collection
gc.enable()

# Create objects
obj1 = Demo("A")
obj2 = Demo("B")

# Create circular reference
obj1.ref = obj2
obj2.ref = obj1

# Remove references
obj1 = None
obj2 = None

# Force garbage collection
print("Collecting garbage...")
collected_objects = gc.collect()

print(f"Number of unreachable objects collected: {collected_objects}")
