cpp
const size_t BLOCK_SIZE = 256;
const size_t MAX_SHARED_POINTS = 1024;

__device__ bool comparePoints(const Point& a, const Point& b) {
    return (a.x < b.x) || ((a.x == b.x) && (a.y < b.y));
}

__global__ void closestPair(const Point *points, size_t numPoints, double *distance) {
    __shared__ Point sharedPoints[MAX_SHARED_POINTS];
    __shared__ double sharedMinDist;
    __shared__ int sharedMinI, sharedMinJ;
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (gid < numPoints) {
        sharedPoints[tid] = points[gid];
    }
    __syncthreads();
    
    if (tid == 0) {
        sharedMinDist = INFINITY;
        sharedMinI = -1;
        sharedMinJ = -1;
    }
    __syncthreads();
    
    for (int i = tid; i < numPoints; i += blockDim.x) {
        Point pi = sharedPoints[i];
        for (int j = i + 1; j < numPoints; j++) {
            Point pj = sharedPoints[j];
            double dx = pj.x - pi.x;
            double dy = pj.y - pi.y;
            double d = sqrt(dx * dx + dy * dy);
            
            if (d < sharedMinDist) {
                sharedMinDist = d;
                sharedMinI = i;
                sharedMinJ = j;
            }
        }
    }
    __syncthreads();
    
    if (tid == 0) {
        *distance = sharedMinDist;
    }
}