import importlib
spam_loader = importlib.util.find_spec('src')
found = spam_loader is not None
    
print(found)