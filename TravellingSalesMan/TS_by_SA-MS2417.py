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
			
			city += ", India"
			pt = geolocator.geocode(city,timeout = 10000)
			print("City = ", city,pt.latitude,pt.longitude)
			P.insert(j,[pt.latitude,pt.longitude])
			j += 1
	return P

P = readcities()
print('no of cities =' ,len(P))

if __name__ == '__main__':
    PNames = []
    
    P = readcities(PNames)
    nCity = len(P)