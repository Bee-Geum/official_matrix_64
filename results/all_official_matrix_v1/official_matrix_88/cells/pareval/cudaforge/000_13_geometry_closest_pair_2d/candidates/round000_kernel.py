cpp
	__shared__ double minDist;
	if (threadIdx.x == 0) {
		minDist = DBL_MAX;
	}
	__syncthreads();

	for (size_t i = threadIdx.x; i < numPoints - 1; i += blockDim.x) {
		for (size_t j = i + 1; j < numPoints; ++j) {
			double dist = distanceBetweenPoints(points[i], points[j]);
			if (dist < minDist) {
				minDist = dist;
			}
		}
	}
	__syncthreads();

	if (threadIdx.x == 0) {
		atomicMin(&minDist, minDist);
		*distance = minDist;
	}
}
