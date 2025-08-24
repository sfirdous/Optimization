from numpy import *
from geopy.geocoders import Nominatim

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
    Pt += [P[seq[0]]]  # close the loop
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
	with open("india_cities.txt") as file:
		for line in file:
			city = line.rstrip('\n')
			if(city == ""):
				break
			
			location += ", India"
			pt = geolocator.geocode(location,timeout = 10000)
			x = round(pt.longitude,2)
			y = round(pt.latitude,2)
			print("City = ", city,pt.latitude,pt.longitude)
			P.insert(j,[x,y])
			PNames.insert(j,city)
			j += 1
	return P

P = readcities()
print('no of cities =' ,len(P))

if __name__ == '__main__':
    PNames = []
    
    P = readcities(PNames)
    nCity = len(P)