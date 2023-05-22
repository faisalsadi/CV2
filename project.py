import numpy as np

def xor(array1,array2):
    result=np.bitwise_xor(array1,array2)
    count=0
    size=result.size
    for i in range(size) :
        if result[i]==1:
            count+=1
    return count


def censusTransform(matrix, k):
    n = matrix.shape[0]
    m = matrix.shape[1]
    k_values = np.zeros((n, m, k*k))
    for i in range(n):
        for j in range(m):
            if i+k//2>=n-1:
                end_row = n-1
                start_row = i - (k - (n - i))

            else:
                 start_row = max(i - k // 2, 0)
                 end_row =i+( k-(i-start_row))-1
            if j+k//2>=m-1:
                end_col = m-1
                start_col = j - (k - (m - j))

            else:
                    start_col = max(j - k // 2, 0)
                    end_col = j+( k-(j-start_col))-1
            submatrix = np.zeros((k, k))
            submatrix= matrix[start_row:end_row+1, start_col:end_col+1].copy()
            for x in range(k):
                for y in range(k):

                    if submatrix[x][y]>=matrix[i][j]:
                        submatrix[x][y]=1
                    else:
                        submatrix[x][y]=0
            victor = submatrix.reshape(k*k)
            k_values[i, j ] = victor

    return k_values



def costVolume(victorL,victorR,file_content):
    n=victorL.shape[0]
    m=victorL.shape[1]
    file_content=int(file_content)
    victor=np.zeros((n,m,file_content))
    for i in range(n):
        for j in range(m):
            for x in range(file_content):
                start_row=max(0,j-file_content)
                for k in range(start_row,j+1):
                    vL=victorL[i][j].astype(int)
                    vR=victorR[i][k].astype(int)
                    count=xor(vL,vR)
                    victor[i][j]=count

    return victor
def disparity(imageL,imageR,k,file_content):

    victorL=censusTransform(imageL,k)
    victorR=censusTransform(imageR,k)
    victorLR=costVolume(victorL,victorR,file_content)
    print(victorLR)

    # print(victorL[0][0])
    # array=np.array(victorL[0][0])
    # print(array)
    # array=array.astype(int)
    # # count=xor(victorL[0][0][:],victorR[0][0][:])
    # # print(count)
    # #print(victorLR)
    # # print(victorL[0][0][0])
    # # print(file_content)
    # array2=np.array(victorR[0][0])
    # array2=array2.astype(int)
    # # array2 = np.array([1,1 ,0 ,1, 0])
    # print(array2)
    # # array = np.array([1,1 ,1 ,1, 0])
    # c=xor(array,array2)