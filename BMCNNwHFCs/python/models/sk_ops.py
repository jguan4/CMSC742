import tensorflow as tf
import tensorly as tl

def mode_n_prod_4(T,U,n):
    n = int(n)
    # We need one letter per dimension
    # (maybe you could find a workaround for this limitation)
    ind = ''.join(chr(ord('a') + i) for i in range(n))
    exp = f'{ind}K...,JK->{ind}J...'
    return tf.einsum(exp, T, U)

def mode_n_prod_T_4(T,U,n):
    n = int(n)
    # We need one letter per dimension
    # (maybe you could find a workaround for this limitation)
    ind = ''.join(chr(ord('a') + i) for i in range(n))
    exp = f'{ind}K...,KJ->{ind}J...'
    return tf.einsum(exp, T, U)

def dot_4(T,U,d2):
	[k,h,w,d1]  = T.shape
	res = tf.zeros([d2,h,w,d1])
	dot_exp = 'ijk,ijk->'
	for s in range(d1):
		for x in range(d2):
			for y in range(h):
				for z in range(w):
					for c in range(k):
						for i in range(h):
							for j in range(w):
								res[x,y,z,s] += T[c,i,j,s]*U[(c+1)*(i+1)*(j+1)-1,(x+1)*(y+1)*(z+1)-1]
	return res

def out_prod_4(veclist):
	[a,b,c,d] = veclist
	e = tf.einsum('i,j,k,m->ijkm', a,b,c,d)
	return e

def out_prod_2(veclist):
	[a,b] = veclist
	e = tf.einsum('i,j->ij', a,b)
	return e

def out_prod_n(veclist):
	n = len(veclist)
	if n == 2:
		return out_prod_2(veclist)
	elif n == 4:
		return out_prod_4(veclist)
