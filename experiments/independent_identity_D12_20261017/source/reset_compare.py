"""Only normalize sequence container representation; retain every field and value."""
def canonical(value):
 if isinstance(value,dict):return {key:canonical(v)for key,v in value.items()}
 if isinstance(value,(tuple,list)):return [canonical(v)for v in value]
 return value

def same_data(a,b):
 return canonical(a)==canonical(b)
