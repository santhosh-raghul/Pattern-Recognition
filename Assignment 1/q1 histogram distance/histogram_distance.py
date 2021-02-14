from math import log2
import math
import numpy as np
 
# calculate the kl divergence
# KL(P || Q) = sum x in X P(x) * log(P(x) / Q(x))
# Where the “||” operator indicates “divergence” or Ps divergence from Q.
def kl_divergence(p, q):
	return sum(p[i] * np.log(p[i]/q[i]) for i in range(len(p)))

# calculate bhattacharya distance
# For probability distributions p and q over the same domain X, the Bhattacharyya distance is defined as
# DB(p,q) = −ln(BC(p ,q))
# where BC(p,q) = sum(sqrt(p[i]*q[i])) for i in range(len(p))
def bhattacharya(p,q):
	sum_x = sum(math.sqrt(p[i]*q[i]) for i in range(len(p)))
	la = - math.log(sum_x)
	return la

if __name__ == "__main__":
	h1 = [ 0.24, 0.2, 0.16, 0.12, 0.08, 0.04, 0.12, 0.04]
	h2 = [ 0.22, 0.23, 0.16, 0.13, 0.11, 0.08, 0.05, 0.02]

	# calculate (H1 || H2)
	print("KL distance :")
	kl_pq = kl_divergence(h1,h2)
	print('\tKL(H1 || H2): %f' % kl_pq)
	# calculate (H2 || H1)
	kl_qp = kl_divergence(h2,h1)
	print('\tKL(H2 || H1): %f' % kl_qp)
	d = bhattacharya(h1,h2)
	print('Bhattacharya distance: %f' % d)