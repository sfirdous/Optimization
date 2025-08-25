from scipy import *
from numpy import *
from pylab import *
from geopy.geocoders import Nominatim
import random

def Distance(P1,P2):
    if P1 == P2:
        return 0.0
    
    d = sqrt((P1[0] - P2[0])** 2 + (P1[1] - P2[1]) ** 2)
    return d

def TotalDistance(P,seq):
    dist = 0.0
    N = len(seq)
    for i in range(N-1):
        dist += Distance(P[seq[i]],P[seq[i+1]])
    dist += Distance(P[seq[N-1]],P[seq[0]]) 
    
    return dist

def Plot(P, seq, dist, PNames):
    Pt = [P[seq[i]] for i in range(len(seq))]
    Pt += [P[seq[0]]]
    Pt = array(Pt)

    title("Total Distance : " + str(dist))
    plot(Pt[:,0], Pt[:,1], '-o')

    for i in range(len(P)):
        annotate(PNames[i], (P[i][0], P[i][1]))

    show()
 
 

def readcities(PNames):
    
	P = []  # coordinates of cities
	j = 0
 
	#Get coordinates of cities
	geolocator = Nominatim(user_agent = "fss_app")
	with open("India_cities.txt") as file:
		for line in file:
			city = line.rstrip('\n')
			if(city == ""):
				break
			
			location = city + ", India"
			pt = geolocator.geocode(location,timeout = 10000)
			x = round(pt.longitude,2)
			y = round(pt.latitude,2)
			print("City = ", city,pt.latitude,pt.longitude)
			P.insert(j,[x,y])
			PNames.insert(j,city)
			j += 1
	return P

def reverse(P,seq,dist,N1,N2,temp,nCity):
    N1L = N1-1
    if(N1L < 0):
        N1L += nCity
    
    N2R = N2 + 1
    if(N2R >= nCity):
        N2R = 0
        
    delta = 0.0
    if(N1 != N2R) and (N2 != N1L):
        delta = Distance(P[seq[N1L]],P[seq[N2]]) + Distance(P[seq[N1]],P[seq[N2R]]) - Distance(P[seq[N1L]], P[seq[N1]]) - Distance(P[seq[N2]],P[seq[N2R]])
    else:
        return dist,False
    
    prob = 1.0
    if(delta > 0.0):
        prob = exp(-delta/temp)
    
    rndm = random.random()
    if(rndm < prob):
        dist += delta
        
        i = N1
        j = N2
        
        while(i < j):
            u = seq[i]
            seq[i] = seq[j]
            seq[j] = u
            i += 1
            j -= 1
        dif = abs(dist - TotalDistance(P,seq))
        if(dif*dist > 0.01):
            print("in REVERSE N1L=%3d N2R=%3d \n" % (N1L, N2R) )
            print( "N1=%3d N2=%3d T= %f D= %f delta= %f p= %f rn= %f\n" %(N1, N2, temp, dist, delta, prob, rndm) )
            print(seq)
            print()
            input("Press Enter to continue...")
        return dist,True
    else:
        return dist,False
    
def swap(P,seq,dist,N1,N2,temp,nCity):
    N1L = N1 - 1
    if(N1L < 0):
        N1L += nCity
        
    N1R = N1 + 1
    if(N1R >= nCity):
        N1R  = 0
    
    N2L = N2 -1
    if(N2L < 0):
        N2L += nCity
    
    N2R = N2 + 1
    if(N2R >= nCity):
        N2R = 0
    
    I1 = seq[N1]
    I2 = seq[N2]
    I1L = seq[N1L]
    I1R = seq[N1R]
    I2L = seq[N2L]
    I2R = seq[N2R]
    
    delta = 0.0
    #Add distances of new edges after swap.
    delta += Distance(P[I1L],P[I2])
    delta += Distance(P[I1],P[I2R])
    
    #Subtract distances of old edges before swap.
    delta -= Distance(P[I1L],P[I1])
    delta -= Distance(P[I2],P[I2R])
    
    
    #cases where swapped cities aren’t next to each other or adjacent neighbors
    if(N1 != N2L and N1R != N2 and N1R != N2L and N2 != N1L):
        delta += Distance(P[I2], P[I1R])
        delta += Distance(P[I2L], P[I1])
        delta -= Distance(P[I1], P[I1R])
        delta -= Distance(P[I2L], P[I2])
        
        
    prob = 1.0          #acceptance probability for the swap
    #(swap worsens tour) jump to find global minima
    if(delta > 0.0):
        prob = exp(-delta/temp)
        
    rndm = random.random()
    if(rndm < prob):
        dist += delta
        seq[N1] = I2
        seq[N2] = I1
        dif = abs(dist - TotalDistance(P, seq))
        if(dif*dist > 0.01):
            print("%s\n" %("in SWAP -->"))
            print( "N1=%3d N2=%3d N1L=%3d N1R=%3d N2L=%3d N2R=%3d \n" % (N1,N2, N1L, N1R, N2L, N2R) )
            print( "I1=%3d I2=%3d I1L=%3d I1R=%3d I2L=%3d I2R=%3d \n" % (I1,I2, I1L, I1R, I2L, I2R) )
            print( "T= %f D= %f delta= %f p= %f rn= %f\n" % (temp, dist,delta, prob, rndm) )
            print(seq)
            print("%s\n" % ("") )
            input("Press Enter to continue...")
        return dist, True
    else:
        return dist, False
        

if __name__ == '__main__':
    PNames = []  # Names of cities
    P = readcities(PNames)  # Read city coordinates and fill city names
    nCity = len(P)  # Number of cities to visit

    maxTsteps = 300         # Number of temperature steps (cooling iterations)
    fCool = 0.85            # Cooling factor (temperature multiplier each step)
    maxSwaps = 2500         # Maximum swap attempts per temperature
    maxAccepted = 20 * nCity  # Number of accepted moves per temperature
    

    seq = arange(0, nCity, 1)  # Initial sequence of city indices
    dist = TotalDistance(P, seq)  # Initial total distance of the tour
    temp = 10.0 * dist  # Starting temperature, must be high enough

    print("\n\n")
    print(seq)
    print("\n nCity= %3d dist= %f temp= %f \n" % (nCity, dist, temp) )
    input("Press Enter to continue...")

    Plot( P,seq, dist, PNames)  # Initial plot of the tour

    oldDist = 0.0
    convergenceCount = 0

    # Cooling loop for simulated annealing
    for t in range(1, maxTsteps + 1):
        if temp < 1.0e-6:  # Stop if temperature is very low
            break

        accepted = 0  # Number of accepted moves at current temperature
        iteration = 0  # Number of iterations tried at current temperature

        # Run swaps/reverses for maxSwaps iterations
        while iteration <= maxSwaps:
            # Select two distinct random city indices N1 and N2
            N1 = -1
            while N1 < 0 or N1 >= nCity:
                N1 = int(random.random() * 1000.0) % nCity

            N2 = -1
            while N2 < 0 or N2 >= nCity or N2 == N1:
                N2 = int(random.random() * 1000.0) % nCity

            # Swap N1 and N2 if N2 is less than N1 (to enforce order)
            if N2 < N1:
                N1, N2 = N2, N1  # Pythonic swap

            # Randomly decide to do swap or reverse operation
            chk = random.uniform(0, 1)

            # Conditions to choose swap or reverse to avoid invalid swaps
            if (chk < 0.5) and (N1 + 1 != N2) and (N1 != ((N2 + 1) % nCity)):
                dist, flag = swap(P, seq, dist, N1, N2, temp, nCity)
            else:
                dist, flag = reverse(P, seq, dist, N1, N2, temp, nCity)

            # Increase accepted count if change accepted
            if flag:
                accepted += 1
            iteration += 1

        # Print status after each cooling step
        print("Iteration: %d temp=%f dist=%f" % (t, temp, dist))
        print("seq = ")
        set_printoptions(precision=3)
        print(seq)
        print("\n\n")

        # Check for convergence if distance does not change much
        if abs(dist - oldDist) < 1.0e-4:
            convergenceCount += 1
        else:
            convergenceCount = 0

        # Stop if solution converged for 4 consecutive temp steps
        if convergenceCount >= 4:
            break

        # Plot intermediate result every 25 temperature updates
        if (t % 25) == 0:
            Plot( P,seq, dist, PNames)

        # Decrease temperature by cooling factor
        temp *= fCool
        oldDist = dist

    # Final plot after simulated annealing completes
    Plot( P,seq, dist, PNames)
