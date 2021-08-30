from olfactory_import import *

def gen_color(dz):
    norm = plt.Normalize()
    colors = plt.cm.jet(norm(dz))

    colors = plt.cm.jet(np.linspace(0,1,len(dz)))

    lower = dz.min()
    upper = dz.max()
    colors = plt.cm.jet((dz-lower)/(upper-lower))

    def get_colors(inp, colormap, vmin=None, vmax=None):
        norm = plt.Normalize(vmin, vmax)
        return colormap(norm(inp))

    colors = get_colors(dz, plt.cm.jet)
    
    return colors
    
def align_z(x,z):
    wgt = np.linalg.lstsq(x-x.mean(), z-z.mean())[0]   
    xx = np.dot(x-x.mean(),wgt)+z.mean()
    return xx
    
def const(x):
    x = tf.constant(x, dtype=tf.float32)
    return x

def var(x):
    x = tf.Variable(x, dtype=tf.float32)
    return x

def softplus_inv(x):
    x = tf.log(tf.exp(const(x))-const(1))
    return x

def rsquared(x, y):
    """ Return R^2 where x and y are array-like."""

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return r_value**2

def logdet(L):
        return 2.0 * np.sum(np.log(np.diag(L)))
    
def rbf_covariance(kernel_params, x, xp):
    output_scale = np.exp(kernel_params[0])
    lengthscales = np.exp(kernel_params[1:])
    diffs = np.expand_dims(x /lengthscales, 1)\
          - np.expand_dims(xp/lengthscales, 0)
    return output_scale * np.exp(-0.5 * np.sum(diffs**2, axis=2))

