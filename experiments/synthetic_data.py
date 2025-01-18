import numpy as np
import scipy as sp
from numpy.typing import NDArray
from statsmodels.tsa.arima_process import ArmaProcess

from robust_deconfounding.utils import cosine_basis, haarMatrix


class BaseDataGenerator:
    def __init__(self, basis_type: str, beta: NDArray, noise_var: float, noise_type="normal") -> None:
        """
        Base class for generating data for deconfounding experiments.

        Attributes:
            basis_type (str): Type of basis to use for transformation ("cosine", "haar", or None).
            beta (NDArray): True causal coefficient vector for the linear relationship.
            noise_var (float): Variance of the noise added to the data.
        """
        if basis_type not in ["cosine", "haar", None]:
            raise ValueError("basis not implemented")
        self.basis_type = basis_type
        self.beta = beta
        self.noise_var = noise_var
        self.noise_type= noise_type

    def get_noise_vars(self, n: int, sizes: list[int]) -> tuple[NDArray, NDArray, NDArray]:
        """
        Generates noise terms for the data with different variances.

        Args:
            n (int): Number of data points.
            sizes (list[int]): Sizes of the noise arrays (for x, y, u).

        Returns:
            tuple[NDArray, NDArray, NDArray]: Noise arrays for x, y, u.
        """
        if self.noise_type=="uniform":
            a=np.sqrt(3*self.noise_var)     #unifrom on the interval [-a,a]
            b=np.sqrt(3/10)
            c=np.sqrt(3)
            noises = [np.random.uniform(-(a if i == 2 else (b if i==0 else c)), (a if i == 2 else (b if i==0 else c)), size=(n, size)) for i, size in enumerate(sizes)]
        elif self.noise_type=="normal":
            b=np.sqrt(3)
            c=np.sqrt(3/10)
            noise_u = np.random.normal(0, 1, size=(n, 1))
            noise_x = np.random.normal(0, 1, size=(n, 1))
            #noise_u=np.random.uniform(-b, b, size=(n,1)) 
            #noise_x= np.random.uniform(-c, c, size=(n,1))
            noise_y = np.random.normal(0, np.sqrt(self.noise_var), size=(n, 1))
            noises=[noise_u, noise_x, noise_y]
        elif self.noise_type=="normal_uniform":
            b=np.sqrt(3)
            c=np.sqrt(3/10)
            noise_u=np.random.uniform(-b, b, size=(n,1)) 
            noise_x= np.random.uniform(-c, c, size=(n,1))
            noise_y = np.random.normal(0, np.sqrt(self.noise_var), size=(n, 1))
            noises=[noise_u, noise_x, noise_y]
        else:
            raise ValueError("Noise Type not implemented.")
        
        return noises

    def get_basis(self, n: int):
        """
        Returns the chosen basis for data transformation.

        Args:
            n (int): Number of data points.

        Returns:
            NDArray: Basis matrix.
        """
        if self.basis_type == "cosine":
            return cosine_basis(n)
        elif self.basis_type == "haar":
            return haarMatrix(n)
        else:
            raise ValueError("basis not implemented")

    @staticmethod
    def basis_transform(u: NDArray, outlier_points: NDArray, basis: NDArray, n: int) -> NDArray:
        """
        Returns a version of 'u' that is sparse in 'basis'.

        Args:
            u (NDArray): The data to be transformed.
            outlier_points (NDArray): Indicator vector for outlier data points.
            basis (NDArray): The basis matrix for transformation (or None for Fourier transform).
            n (int): Number of data points.

        Returns:
            NDArray: Transformed data.
        """
        if basis is None:
            un = sp.fft.fft(u.T, norm="forward").T
            k = sp.fft.ifft(un * outlier_points, norm="forward")
        else:
            un = basis.T @ u / n
            k = basis @ (un * outlier_points)
        return k

    def generate_data(self, n: int, outlier_points: NDArray) -> tuple[NDArray, NDArray]:
        """
        Abstract method to be implemented by subclasses to generate data.
        """
        raise NotImplementedError
    
    def generate_data_2_dim(self, n: int, outlier_points: NDArray):
        """
        Abstract method to be implemented by subclasses to generate data.
        """
        raise NotImplementedError


class OUDataGenerator(BaseDataGenerator):
    def __init__(self, basis_type: str, beta: NDArray, noise_var: float):
        super().__init__(basis_type, beta, noise_var)

    def generate_data(self, n: int, outlier_points: NDArray):
        """
        Generates data from discretized Ornstein-Uhlenbeck processes with confounding.

        Args:
            n (int): Number of data points.
            outlier_points (NDArray): Indicator vector for outlier data points.

        Returns:
            tuple[NDArray, NDArray]: The generated data (x, y).
        """
        AR_object1, AR_object2 = self.get_ar(n)

        eu, ex, ey = self.get_noise_vars(n, [1, 1, 1])

        u = AR_object1.generate_sample(nsample=2 * n)[n:2 * n].reshape(-1, 1) + eu
        
        basis = self.get_basis(n)

        k = self.basis_transform(u, outlier_points, basis, n)

        x = AR_object2.generate_sample(nsample=2 * n)[n:2 * n].reshape(-1, 1) + ex + u

        y = x @ self.beta + ey + 10 * k

        return x, y
    
    def generate_data_2_dim(self, n: int, outlier_points: NDArray) -> tuple[NDArray, NDArray]:
        """
        Generates 2-dimensional data from discretized Ornstein-Uhlenbeck processes with confounding.

        Args:
            n (int): Number of data points.
            outlier_points (NDArray): Indicator vector for outlier data points.

        Returns:
            tuple[NDArray, NDArray]: The generated data (x, y).
        """
        AR_object1, AR_object2 = self.get_ar(n)

        eu, ex, ey = self.get_noise_vars(n, [1, 2, 1])

        u = AR_object1.generate_sample(nsample=2 * n)[n:2 * n].reshape(-1, 1) + eu

        basis = self.get_basis(n)
        k = self.basis_transform(u, outlier_points, basis, n)

        x_1 = AR_object2.generate_sample(nsample=2 * n)[n:2 * n].reshape(-1, 1) + u
        x_2 = AR_object2.generate_sample(nsample=2 * n)[n:2 * n].reshape(-1, 1) - 2 * u
        x = np.hstack((x_1, x_2)) + ex
        x = x.reshape(-1, 2)

        y = x @ self.beta + ey + 10 * k
        
        return x, y

    def get_ar(self, n: int) -> tuple[ArmaProcess, ArmaProcess]:
        """
        Generates two discretized Ornstein-Uhlenbeck processes. Note that discretized Ornstein-Uhlenbeck processes are
        AR processes.

        Args:
            n (int): Number of data points.

        Returns:
            tuple[ArmaProcess, ArmaProcess]: Two AR models representing the processes.
        """
        ar1 = np.array([1, -0.8/n])
        ma1 = np.array([1/np.sqrt(n)])
        AR_object1 = ArmaProcess(ar1, ma1)

        ar2 = np.array([1, -0.5/n])
        ma2 = np.array([1/np.sqrt(n)])
        AR_object2 = ArmaProcess(ar2, ma2)

        return AR_object1, AR_object2


class BLPDataGenerator(BaseDataGenerator):
    """
    Generates data with confounding concentrated in a specific frequency band.

    Attributes:
        band (list[int]): Frequency band for concentrated confounding (inclusive).
    """
    def __init__(self, basis_type: str, beta: NDArray, noise_var: float, band: list[int]):
        super().__init__(basis_type, beta, noise_var)
        self.band = band

    def get_band_idx(self, n: int) -> NDArray:
        """
        Generates an indicator vector for the specified frequency band.
        """
        return np.array([1 if i in self.band else 0 for i in range(n)]).reshape(-1, 1)
    
    def generate_data(self, n: int, outlier_points: NDArray) -> tuple[NDArray, NDArray]:
        """
        Generates data from discretized band-limited processes with confounding.

        Args:
            n (int): Number of data points.
            outlier_points (NDArray): Indicator vector for outlier data points.

        Returns:
            tuple[NDArray, NDArray]: The generated data (x, y).
        """
        eu, ex, ey = self.get_noise_vars(n, [1, 1, 1])
        band_idx = self.get_band_idx(n)

        basis = self.get_basis(n)

        weights = np.random.normal(0, 1, size=(n, 1))
        u_band = basis @ (weights * band_idx)
        u = u_band + eu

        weights = np.random.normal(0, 1, size=(n, 1))
        x_band = basis @ (weights * band_idx)
        x = x_band + u + ex

        k = self.basis_transform(u, outlier_points, basis, n)

        y = x @ self.beta + ey + 10 * k
        
        return x, y
    
    def generate_data_2_dim(self, n: int, outlier_points: NDArray) -> tuple[NDArray, NDArray]:
        """
        Generates data from 2-dimensional discretized band-limited processes with confounding.

        Args:
            n (int): Number of data points.
            outlier_points (NDArray): Indicator vector for outlier data points.

        Returns:
            tuple[NDArray, NDArray]: The generated data (x, y).
        """
        eu, ex, ey = self.get_noise_vars(n, [1, 2, 1])
        band_idx = self.get_band_idx(n)

        basis = self.get_basis(n)

        weights = np.random.normal(0, 1, size=(n, 1))
        u_band = basis @ (weights * band_idx)
        u = u_band + eu

        weights = np.random.normal(0, 1, size=(n, 1))
        x_1_band = basis @ (weights * band_idx)
        x_1 = x_1_band + u

        weights = np.random.normal(0, 1, size=(n, 1))
        x_2_band = basis @ (weights * band_idx)
        x_2 = x_2_band - 2 * u

        k = self.basis_transform(u, outlier_points, basis, n)

        x = np.hstack((x_1, x_2)) + ex
        x = x.reshape(-1, 2)

        y = x @ self.beta + ey + 10*k

        return x, y

class BLPNonlinearDataGenerator(BaseDataGenerator):
    """
    Generates data with confounding concentrated in a specific frequency band.
    The relation between x and y is assumued to be nonlinear and specified by the integer beta
    Attributes:
        band (list[int]): Frequency band for concentrated confounding (inclusive).
    """
    def __init__(self, basis_type: str, beta: NDArray, noise_var: float, band: list[int], noise_type="uniform"):
        super().__init__(basis_type, beta, noise_var, noise_type)
        self.band = band

    def get_band_idx(self, n: int) -> NDArray:
        """
        Generates an indicator vector for the specified frequency band.
        """
        return np.array([1 if i in self.band else 0 for i in range(n)]).reshape(-1, 1)

    
    def get_noise_vars(self, n: int, sizes: list[int]) -> tuple[NDArray, NDArray, NDArray]:
        """
        Generates noise terms for the data with different variances.

        Args:
            n (int): Number of data points.
            sizes (list[int]): Sizes of the noise arrays (for x, y, u).

        Returns:
            tuple[NDArray, NDArray, NDArray]: Noise arrays for x, y, u.
        """
        if self.noise_type=="uniform":
            a=np.sqrt(3*self.noise_var)     #unifrom on the interval [-a,a]
            b=np.sqrt(3/10)
            c=np.sqrt(3)
            noises = [np.random.uniform(-(a if i == 2 else (b if i==0 else c)), (a if i == 2 else (b if i==0 else c)), size=(n, size)) for i, size in enumerate(sizes)]
        elif self.noise_type=="normal":
            b=np.sqrt(3)
            c=np.sqrt(3/10)
            noise_u = np.random.normal(0, 1, size=(n, 1))
            noise_x = np.random.normal(0, 1, size=(n, 1))
            noise_y = np.random.normal(0, np.sqrt(self.noise_var), size=(n, 1))
            noises=[noise_u, noise_x, noise_y]
        elif self.noise_type=="normal_uniform":
            b=np.sqrt(3)
            c=np.sqrt(3/10)
            noise_u=np.random.uniform(-b, b, size=(n,1)) 
            noise_x= np.random.uniform(-c, c, size=(n,1))
            noise_y = np.random.normal(0, np.sqrt(self.noise_var), size=(n, 1))
            noises=[noise_u, noise_x, noise_y]
        else:
            raise ValueError("Noise Type not implemented.")
        
        return noises

   
    def generate_data(self, n: int, outlier_points: NDArray) -> tuple[NDArray, NDArray]:
        """
        Generates data from discretized band-limited processes with confounding.

        Args:
            n (int): Number of data points.
            outlier_points (NDArray): Indicator vector for outlier data points.

        Returns:
            tuple[NDArray, NDArray]: The generated data (x, y).
        """ 
        eu, ex, ey = self.get_noise_vars(n, [1, 1, 1])
        band_idx = self.get_band_idx(n)

        basis = self.get_basis(n)

        weights = np.random.normal(0, 1, size=(n, 1))
        u_band = basis @ (weights * band_idx)
        u = u_band + eu
        k = self.basis_transform(u, outlier_points, basis, n)

        weights = np.random.normal(0, 1, size=(n, 1))
        x_band = basis @ (weights * band_idx)
        x = x_band + 2*u + ex
     
        max_k=np.max(k)
        min_k=np.min(k)
        diff_k=max_k-min_k
        k=k/diff_k
        
        max=np.max(x)
        min=np.min(x)
        x=(x-min)/(max-min)

        y = functions_nonlinear(x, self.beta[0]) + ey + 20*k

        return x[:,0], y[:,0], 20*k[:,0]


class OUNonlinearDataGenerator(BaseDataGenerator):
    def __init__(self, basis_type: str, beta: NDArray, noise_var: float, noise_type="normal"):
        super().__init__(basis_type, beta, noise_var, noise_type)

    def generate_data(self, n: int, outlier_points: NDArray):
        """
        Generates data from discretized Ornstein-Uhlenbeck processes with confounding.

        Args:
            n (int): Number of data points.
            outlier_points (NDArray): Indicator vector for outlier data points.

        Returns:
            tuple[NDArray, NDArray]: The generated data (x, y).
        """
        AR_object1, AR_object2 = self.get_ar(n)

        eu, ex, ey = self.get_noise_vars(n, [1, 1, 1])

        u = AR_object1.generate_sample(nsample=2 * n)[n:2 * n].reshape(-1, 1) + eu
        
        basis = self.get_basis(n)

        k = self.basis_transform(u, outlier_points, basis, n)

        x = AR_object2.generate_sample(nsample=2 * n)[n:2 * n].reshape(-1, 1)  + 2*k + ex  

        y = functions_nonlinear(x, self.beta[0]) + ey + 10*k

        return  x[:,0], y[:,0], 10*k[:,0]
    

    def get_ar(self, n: int) -> tuple[ArmaProcess, ArmaProcess]:
        """
        Generates two discretized Ornstein-Uhlenbeck processes. Note that discretized Ornstein-Uhlenbeck processes are
        AR processes.

        Args:
            n (int): Number of data points.

        Returns:
            tuple[ArmaProcess, ArmaProcess]: Two AR models representing the processes.
       
        """
        ar1 = np.array([1, -0.8/n])
        ma1 = np.array([1/np.sqrt(n)])
        AR_object1 = ArmaProcess(ar1, ma1)

        ar2 = np.array([1, -0.5/n])
        ma2 = np.array([1/np.sqrt(n)])
        AR_object2 = ArmaProcess(ar2, ma2)

        return AR_object1, AR_object2

    def get_noise_vars(self, n: int, sizes: list[int]) -> tuple[NDArray, NDArray, NDArray]:
        """
        Generates noise terms for the data with different variances.

        Args:
            n (int): Number of data points.
            sizes (list[int]): Sizes of the noise arrays (for x, y, u).

        Returns:
            tuple[NDArray, NDArray, NDArray]: Noise arrays for x, y, u.
        """
        if self.noise_type=="uniform":
            a=np.sqrt(3*self.noise_var)     #unifrom on the interval [-a,a]
            b=np.sqrt(3/10)
            c=np.sqrt(3)
            noises = [np.random.uniform(-(a if i == 2 else (b if i==0 else c)), (a if i == 2 else (b if i==0 else c)), size=(n, size)) for i, size in enumerate(sizes)]
        elif self.noise_type=="normal":
            b=np.sqrt(3)
            c=np.sqrt(3/10)
            noise_u = np.random.normal(0, 1, size=(n, 1))
            noise_x = np.random.normal(0, 1, size=(n, 1))
            noise_y = np.random.normal(0, np.sqrt(self.noise_var), size=(n, 1))
            noises=[noise_u, noise_x, noise_y]
        elif self.noise_type=="normal_uniform":
            b=np.sqrt(3)
            c=np.sqrt(3/10)
            noise_u=np.random.uniform(-b, b, size=(n,1)) 
            noise_x= np.random.uniform(-c, c, size=(n,1))
            noise_y = np.random.normal(0, np.sqrt(self.noise_var), size=(n, 1))
            noises=[noise_u, noise_x, noise_y]
        else:
            raise ValueError("Noise Type not implemented.")
        
        return noises  
    
class UniformNonlinearDataGenerator(BaseDataGenerator):
    """
    Generates data with confounding concentrated in a specific frequency band.
    The relation between x and y is assumued to be nonlinear and specified by the integer beta
    Attributes:
        band (list[int]): Frequency band for concentrated confounding (inclusive).
    """
    def __init__(self, basis_type: str, beta: NDArray, noise_var: float,  noise_type="uniform"):
        super().__init__(basis_type, beta, noise_var, noise_type)
    
    def get_noise_vars(self, n: int, sizes: list[int]) -> tuple[NDArray, NDArray, NDArray]:
        """
        Generates noise terms for the data with different variances.

        Args:
            n (int): Number of data points.
            sizes (list[int]): Sizes of the noise arrays (for x, y, u).

        Returns:
            tuple[NDArray, NDArray, NDArray]: Noise arrays for x, y, u.
        """

        noise_u=np.random.uniform(0, 1, size=(n,1)) 
        noise_x= np.random.uniform(0, 1, size=(n,1))
        noise_y = np.random.normal(0, np.sqrt(self.noise_var), size=(n, 1))
        noises=[noise_u, noise_x, noise_y]

        return noises

   
    def generate_data(self, n: int, outlier_points: NDArray) -> tuple[NDArray, NDArray]:
        """
        Generates data from processes with x indepent, indentical unifrom distributed with confounding.

        Args:
            n (int): Number of data points.
            outlier_points (NDArray): Indicator vector for outlier data points.

        Returns:
            tuple[NDArray, NDArray]: The generated data (x, y).
        """ 
        eu, ex, ey = self.get_noise_vars(n, [1, 1, 1])

        basis = self.get_basis(n)

        x=ex
        k = self.basis_transform(6*x-3, outlier_points, basis, n)

        y = functions_nonlinear(x, self.beta[0]) + ey + 10*k

        return x[:,0], y[:,0], 10*k[:,0]
    

class OUReflectedNonlinearDataGenerator(OUNonlinearDataGenerator):
    def __init__(self, basis_type: str, beta: NDArray, noise_var: float, noise_type="normal"):
        super().__init__(basis_type, beta, noise_var, noise_type)

    def generate_data(self, n: int, outlier_points: NDArray):
        """
        Generates data from discretized Ornstein-Uhlenbeck processes with confounding.

        Args:
            n (int): Number of data points.
            outlier_points (NDArray): Indicator vector for outlier data points.

        Returns:
            tuple[NDArray, NDArray]: The generated data (x, y).
        """
        AR_object1, AR_object2 = self.get_ar(n)

        eu, ex, ey = self.get_noise_vars(n, [1, 1, 1])

        u = AR_object1.generate_sample(nsample=2 * n)[n:2 * n].reshape(-1, 1) + eu
        u=self.reflect(u)
        
        basis = self.get_basis(n)
        k = self.basis_transform(u, outlier_points, basis, n)

        x = AR_object2.generate_sample(nsample=2 * n)[n:2 * n].reshape(-1, 1)  + 2*k + ex
        x=self.refelct(x)  

        y = functions_nonlinear(x, self.beta[0]) + ey + 10*k

        return  x[:,0], y[:,0], 10*k[:,0]
    
    def reflect(self, x:NDArray)->NDArray:
        """
        Reflects the proccess x on the boundaries 0, 1.
        Args:
            x (NDArray): Process to reflect
        """

        n=len(x)
        for i in range(0,n):
            if x[i]>=1:
                x[i:n]=2-x[i:n]
            elif x[i]<=0:
                x[i:n]=-x[i:n]
        return x



def functions_nonlinear(x:NDArray, beta:int):
    """"
    Returns the value of different nonlinear functions evaluated at x where the type can be choosen over the integer beta.
    1: Quadratic
    2: Sinfunction
    3: Sigmoid
    4: Finite expansion in function basis
    5: Linear
    """
  
    if beta==1:
        y = 5*x**2 
    elif beta==2:
        y = 6*np.sin(2*np.pi*x)
    elif beta==3:
        y=5/(1+np.exp(-16*x+6))-2.5
    elif beta==4:
        y=-1+np.cos(np.pi*x)-3*np.cos(2*np.pi*x)
    elif beta==5:
        y=3*x
    else:
        raise ValueError("Function not implemented.")
    return y