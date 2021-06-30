import tensorflow as tf

__all__ = ["cca_loss"]


def cca_loss(outdim_size, use_all_singular_values):
    """
    The main loss function (inner_cca_objective) is wrapped in this function due to
    the constraints imposed by Keras on objective functions
    """

    def inner_cca_objective(y_true, y_pred):
        """
        It is the loss function of CCA as introduced in the original paper. There can be other formulations.
        It is implemented on Tensorflow based on github@VahidooX's cca loss on Theano.
        y_true is just ignored
        """

        r1 = 1e-4
        r2 = 1e-4
        eps = 1e-12
        o1 = o2 = int(y_pred.shape[1] // 2)

        # unpack (separate) the output of networks for view 1 and view 2
        H1 = tf.transpose(a=y_pred[:, 0:o1])
        H2 = tf.transpose(a=y_pred[:, o1: o1 + o2])

        m = tf.shape(input=H1)[1]

        H1bar = H1 - tf.cast(tf.divide(1, m), tf.float32) * tf.matmul(
            H1, tf.ones([m, m])
        )
        H2bar = H2 - tf.cast(tf.divide(1, m), tf.float32) * tf.matmul(
            H2, tf.ones([m, m])
        )

        SigmaHat12 = tf.cast(tf.divide(1, m - 1), tf.float32) * tf.matmul(
            H1bar, H2bar, transpose_b=True
        )  # [dim, dim]
        SigmaHat11 = tf.cast(tf.divide(1, m - 1), tf.float32) * tf.matmul(
            H1bar, H1bar, transpose_b=True
        ) + r1 * tf.eye(o1)
        SigmaHat22 = tf.cast(tf.divide(1, m - 1), tf.float32) * tf.matmul(
            H2bar, H2bar, transpose_b=True
        ) + r2 * tf.eye(o2)

        # Calculating the root inverse of covariance matrices by using eigen decomposition
        [D1, V1] = tf.linalg.eigh(SigmaHat11)
        [D2, V2] = tf.linalg.eigh(SigmaHat22)  # Added to increase stability

        posInd1 = tf.compat.v1.where(tf.greater(D1, eps))
        D1 = tf.gather_nd(D1, posInd1)  # get eigen values that are larger than eps
        V1 = tf.transpose(
            a=tf.nn.embedding_lookup(params=tf.transpose(a=V1), ids=tf.squeeze(posInd1))
        )

        posInd2 = tf.compat.v1.where(tf.greater(D2, eps))
        D2 = tf.gather_nd(D2, posInd2)
        V2 = tf.transpose(
            a=tf.nn.embedding_lookup(params=tf.transpose(a=V2), ids=tf.squeeze(posInd2))
        )

        SigmaHat11RootInv = tf.matmul(
            tf.matmul(V1, tf.linalg.tensor_diag(D1 ** -0.5)), V1, transpose_b=True
        )  # [dim, dim]
        SigmaHat22RootInv = tf.matmul(
            tf.matmul(V2, tf.linalg.tensor_diag(D2 ** -0.5)), V2, transpose_b=True
        )

        Tval = tf.matmul(tf.matmul(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)

        if use_all_singular_values:
            corr = tf.sqrt(tf.linalg.trace(tf.matmul(Tval, Tval, transpose_a=True)))
        else:
            [U, V] = tf.linalg.eigh(tf.matmul(Tval, Tval, transpose_a=True))
            U = tf.gather_nd(U, tf.compat.v1.where(tf.greater(U, eps)))
            kk = tf.reshape(tf.cast(tf.shape(input=U), tf.int32), [])
            K = tf.minimum(kk, outdim_size)
            w, _ = tf.nn.top_k(U, k=K)
            corr = tf.reduce_sum(input_tensor=tf.sqrt(w))

        return -corr

    return inner_cca_objective
