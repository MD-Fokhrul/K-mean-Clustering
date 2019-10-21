import numpy as np 
import matplotlib.pyplot as plt

# incomplete in elbow rule 

def kmeans(data, k=3, normalize=False, limit=5000):
    """Basic k-means clustering algorithm.
    """
    # optionally normalize the data. k-means will perform poorly or strangely if the dimensions
    # don't have the same ranges.
    if normalize:
        stats = (data.mean(axis=0), data.std(axis=0))
        data = (data - stats[0]) / stats[1]
    
    # pick the first k points to be the centers. this also ensures that each group has at least
    # one point.
    #centers = data[:k]
    centers = data[np.random.choice(np.arange(len(data)), k), :]

    for i in range(limit):
        # core of clustering algorithm...
        # first, use broadcasting to calculate the distance from each point to each center, then
        # classify based on the minimum distance.
        classifications = np.argmin(((data[:, :, None] - centers.T[None, :, :])**2).sum(axis=1), axis=1)
        '''
        for example if we have data matrix 200x2. and have k = 5 and matrix initialize look like 5x2 matrix
        how can we subtract ? 
        we just transform matrix 200x2 --> matrix 200x2x1
        and centers matrix 5x2 --> centers matrix 1X5x2 subtract them......... we got 200x2x5 tensor !!!
        subtract each point of data to each center. technically, we then square it. and get argmin with axis = 1
        thats our output !
        '''
        # next, calculate the new centers for each cluster.
        new_centers = np.array([data[classifications == j, :].mean(axis=0) for j in range(k)])

        # if the centers aren't moving anymore it is time to stop.
        
                    
        if (new_centers == centers).all():
            break
        else:
            centers = new_centers
    else:
        # this will not execute if the for loop exits on a break.
        raise RuntimeError("Clustering algorithm did not complete within {0:d} iterations".format(limit))
            
    # if data was normalized, the cluster group centers are no longer scaled the same way the original
    # data is scaled.
    
    #WCSS_array=np.append(WCSS_array,kmeans.WCSS())
    if normalize:
        centers = centers * stats[1] + stats[0]
        
    print("Clustering completed after {0:d} iterations".format(i))
    wcss=0
    for j in range(k):
        wcss+=np.sum((classifications[j+1] - new_centers[j,:])**2)
    return {"classifications":classifications , "centers":centers, "wcss" : wcss}  # , "wcss" : wcss
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

#def wcss(k_range = 10 ):
WCSS = []
for i in range(1, 11):
    kmens = kmeans(data = X, k = i)

    
    wcss = kmens['wcss']
    WCSS.append(wcss)
    #return WCSS_array
    

plt.plot(range(1, len(WCSS)+1), WCSS)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmens = kmeans(data = X, k = 5)

center = kmens['centers']
levels = kmens['classifications']

#WCSS = wcss()
      


plt.scatter(X[:, 0], X[:, 1], label = 'Cluster')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()


color=['red','blue','green','cyan','magenta']
clusters=['cluster1','cluster2','cluster3','cluster4','cluster5']
for i in range(5):
    plt.scatter(X[levels == i, 0], X[levels == i, 1], s = 100, c = color[i], label = clusters[i])
plt.scatter(X[levels == 1, 0], X[levels == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[levels == 2, 0], X[levels == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[levels == 3, 0], X[levels == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[levels == 4, 0], X[levels == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(center[:, 0], center[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
