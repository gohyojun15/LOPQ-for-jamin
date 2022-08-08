import torch.nn as nn
import torch
from scipy.cluster.vq import kmeans2, vq

class PQ(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.M, self.Ks = self.opt.M, self.opt.Ks

    @torch.no_grad()
    def fit(self, vecs):
        b, D = vecs.shape
        self.Ds = int(D / self.M)  # dimension of each subspace

        # [m][Ks][Ds]
        self.codewords = torch.zeros((self.M, self.Ks, self.Ds), dtype=torch.float32)
        sub_vecs_splited = torch.split(vecs, split_size_or_sections=self.Ds, dim=1)

        for m, sub_vecs in enumerate(sub_vecs_splited):
            cluster_centers, _ = kmeans2(sub_vecs, self.Ks, iter=50, minit="points")
            self.codewords[m] = torch.from_numpy(cluster_centers)

    @torch.no_grad()
    def encode(self, vecs):
        sub_vecs_splited = torch.split(vecs, split_size_or_sections=self.Ds, dim=1)
        b, D = vecs.shape
        codes = torch.empty((b, self.M), dtype=torch.int64)
        for m, sub_vecs in enumerate(sub_vecs_splited):
            code, value = vq(sub_vecs, self.codewords[m])
            codes[:, m] = torch.from_numpy(code)
        return codes

    @torch.no_grad()
    def decode(self, codes):
        N, M = codes.shape
        vecs = torch.empty((N, self.Ds * self.M), dtype=torch.float32)
        for m in range(self.M):
            vecs[:, m * self.Ds : (m + 1) * self.Ds] = self.codewords[m][codes[:, m], :]
        return vecs

    # def dtable(self, query_vecs):
    #     dtable = torch.empty((self.M, self.Ks), dtype=torch.float32)
    #     subquery = torch.split(query_vecs, split_size_or_sections=self.Ds, dim=0)
    #     for m, query in enumerate(subquery):
    #         dtable[m, :] = torch.norm(self.codewords[m] - query, dim=1, p=2) ** 2
    #     return DistanceTable(dtable)
    @torch.no_grad()
    def dtable(self, query_vecs):
        b, _ = query_vecs.shape

        dtable = torch.empty((b, self.M, self.Ks), dtype=torch.float32)
        split_query = torch.split(query_vecs, split_size_or_sections=self.Ds, dim=1)
        for m, sub_vec_of_query in enumerate(split_query):
            dtable[:, m, :] = torch.norm(sub_vec_of_query.unsqueeze(1) - self.codewords[m].unsqueeze(0), p=2, dim=-1) ** 2
        return DistanceTable(dtable)

class DistanceTable(object):
    def __init__(self, dtable):
        self.dtable = dtable

    @torch.no_grad()
    def adist(self, codes):
        N, M = codes.shape
        dists = torch.sqrt(torch.sum(self.dtable[:, range(M), codes].cuda(), dim=2))
        return dists



class OPQ(object):
    """Pure python implementation of Optimized Product Quantization (OPQ) [Ge14]_.
    OPQ is a simple extension of PQ.
    The best rotation matrix `R` is prepared using training vectors.
    Each input vector is rotated via `R`, then quantized into PQ-codes
    in the same manner as the original PQ.
    .. [Ge14] T. Ge et al., "Optimized Product Quantization", IEEE TPAMI 2014
    Args:
        M (int): The number of sub-spaces
        Ks (int): The number of codewords for each subspace (typically 256, so that each sub-vector is quantized
            into 256 bits = 1 byte = uint8)
        verbose (bool): Verbose flag
    Attributes:
    """

    def __init__(self, opt, verbose=True):
        self.opt = opt
        self.M, self.Ks = self.opt.M, self.opt.Ks
        self.pq = PQ(opt)

        self.R = None

    def fit(self, vecs, pq_iter=20, rotation_iter=10, seed=123):
        """Given training vectors, this function alternatively trains
        (a) codewords and (b) a rotation matrix.
        The procedure of training codewords is same as :func:`PQ.fit`.
        The rotation matrix is computed so as to minimize the quantization error
        given codewords (Orthogonal Procrustes problem)
        This function is a translation from the original MATLAB implementation to that of python
        http://kaiminghe.com/cvpr13/index.html
        If you find the error message is messy, please turn off the verbose flag, then
        you can see the reduction of error for each iteration clearly
        Args:
            vecs (np.ndarray): Training vectors with shape=(N, D) and dtype=np.float32.
            pq_iter (int): The number of iteration for k-means
            rotation_iter (int): The number of iteration for learning rotation
            seed (int): The seed for random process
        Returns:
            object: self
        """
        _, D = vecs.shape
        self.Ds = int(D / self.M)
        self.R = torch.eye(D)

        for i in range(rotation_iter):
            X = vecs @ self.R

            # (a) Train codewords
            pq_tmp = PQ(self.opt)
            if i == rotation_iter - 1:
                # In the final loop, run the full training
                pq_tmp.fit(X)
            else:
                # During the training for OPQ, just run one-pass (iter=1) PQ training
                pq_tmp.fit(X)

            # (b) Update a rotation matrix R
            X_ = pq_tmp.decode(pq_tmp.encode(X))
            U, s, V = torch.linalg.svd(vecs.T @ X_)
            if i == rotation_iter - 1:
                self.pq = pq_tmp
                break
            else:
                self.R = U @ V
        self.codewords = pq_tmp.codewords
        return self

    def rotate(self, vecs):
        """Rotate input vector(s) by the rotation matrix.`
        Args:
            vecs (np.ndarray): Input vector(s) with dtype=np.float32.
                The shape can be a single vector (D, ) or several vectors (N, D)

        """
        if vecs.ndim == 2:
            return vecs @ self.R
        elif vecs.ndim == 1:
            return (vecs.reshape(1, -1) @ self.R).reshape(-1)

    def encode(self, vecs):
        """Rotate input vectors by :func:`OPQ.rotate`, then encode them via :func:`PQ.encode`.
        Args:
            vecs (np.ndarray): Input vectors with shape=(N, D) and dtype=np.float32.
        """
        return self.pq.encode(self.rotate(vecs))

    def decode(self, codes):
        """Given PQ-codes, reconstruct original D-dimensional vectors via :func:`PQ.decode`,
        and applying an inverse-rotation.
        Args:
            codes (np.ndarray): PQ-cdoes with shape=(N, M) and dtype=self.code_dtype.
                Each row is a PQ-code
        """
        # Because R is a rotation matrix (R^t * R = I), R^-1 should be R^t
        return self.pq.decode(codes) @ self.R.T

    def dtable(self, query):
        """Compute a distance table for a query vector. The query is
        first rotated by :func:`OPQ.rotate`, then DistanceTable is computed by :func:`PQ.dtable`.
        Args:
            query (np.ndarray): Input vector with shape=(D, ) and dtype=np.float32
        Returns:
            nanopq.DistanceTable:
                Distance table. which contains
                dtable with shape=(M, Ks) and dtype=np.float32
        """
        return self.pq.dtable(self.rotate(query))