import sys
import math

def simplify_n(spells):
    n = []
    for s in spells:
        for s2 in spells:
            if s == s2:
                continue
            if s % s2 != 0:
                if s not in n:
                    n.append(s)
    return n

def death_n(lower_bound, upper_bound, spells):
    total = 0
    for s in spells:
        multiples = math.floor(max(upper_bound/s - (lower_bound-1)/s, 0))
        total += multiples

    duplicates = 0
    for i in range(len(spells)):
        for j in range(len(spells)-i-1):
            duplicates += find_duplicates(spells[i], spells[i+(j+1)], lower_bound, upper_bound)
    
    return total - duplicates

# Function to count the numbe of grey tiles  
def find_duplicates(x, y, l, r) : 
  
    lcm = (x * y) // math.gcd(x, y) 
  
    # Number multiple of lcm less than L  
    count1 = (l - 1) // lcm 
  
    # Number of multiples of lcm less than R+1  
    countr = r // lcm 
  
    return countr - count1 

def main(argv):
    lower_bound = int(argv[0])
    upper_bound = int(argv[1])
    m = int(argv[2])
    n = list(map(int, argv[3:]))

    if 1 in n:
        print(0)
        return

    deaths = death_n(lower_bound, upper_bound, n)
    print((upper_bound-lower_bound+1) - deaths)        

if __name__ == "__main__":
    main(sys.argv[1:])