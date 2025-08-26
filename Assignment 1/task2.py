def factorial(n):
    result = 1
    for i in range(2, (n+1)):
        result = result * i
    return result

def factorial_recursive(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial_recursive(n-1)

if __name__ == "__main__":
    num = 10
    print(f"The factorial of {num} is {factorial(num)}")
    print(f"The recursive factorial of {num} is {factorial_recursive(num)}")