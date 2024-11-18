import time

def calculate_time(func):
    def inner1(*args, **kwargs):
        begin = time.time()
        if func.__name__ == "runModel":
            print(args[3:6])
        x = func(*args, **kwargs)
        end = time.time()
        print("Total time taken in seconds: ", func.__name__, end - begin)
        return x
    return inner1
