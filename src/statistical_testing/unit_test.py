from src.statstest import *
import pytest

class TestNormal:
    @pytest.fixture
    def arr_list(self):
        return [i for i in range(1, 1001)]
    
    @pytest.fixture
    def arr_normal(self):
        np.random.seed(14)
        return np.random.normal(0, 1, 1000)

    @pytest.fixture(params = [
        (np.random.exponential, (1, 1000)),
        (np.random.poisson, (1, 1000)),
        (np.random.uniform, (0, 1, 1000)),
        (np.random.binomial, (10, 0.5, 1000)),
        (np.random.gamma, (1, 1, 1000))
    ])
    def arr_nonnormal(self, request):
       np.random.seed(14)
       dist_func, dist_args = request.param
       return dist_func(*dist_args)
    
    def test_input_list(self, arr_list):
        alpha = 0.05
        with pytest.raises(TypeError):
            normal_dist_test(arr_list, alpha = alpha)  

    def test_normal(self, arr_normal):
        alpha = 0.05
        result = normal_dist_test(arr_normal, alpha = alpha)
        assert result.normal == True
        assert result.pvalue > alpha

    def test_nonnormal(self, arr_nonnormal):
        alpha = 0.05
        result = normal_dist_test(arr_nonnormal, alpha = alpha)
        assert result.normal == False
        assert result.pvalue <= alpha

class TestEqualVar:
    @pytest.fixture
    def arg_set(self):
        return {i for i in range(1,1001)}
    
    @pytest.fixture
    def args_normal(self):
        np.random.seed(14)
        normal1 = np.random.normal(0, 1, 1000)
        normal2 = np.random.normal(0, 1, 1000)
        normal3 = np.random.normal(10, 100, 1000)
        return normal1, normal2, normal3
    
    def test_input_wrong_argsize(self, args_normal):
        alpha = 0.05
        normal1 = args_normal[0]
        with pytest.raises(ValueError):
            equal_variance_test(normal = True, alpha = alpha)
        with pytest.raises(ValueError):
            equal_variance_test(normal1, normal = True, alpha = alpha)
    
    def test_input_wrong_argtype(self, args_normal, arg_set):
        alpha = 0.05
        normal1, normal2 = args_normal[0:2]
        with pytest.raises(TypeError):
            equal_variance_test(normal1, arg_set, normal = True, alpha = alpha)
        with pytest.raises(TypeError):
            equal_variance_test(normal1, normal2, arg_set, normal = True, alpha = alpha)

    def test_equal_variance(self, args_normal):
        alpha = 0.05
        normal1, normal2 = args_normal[0:2]
        result = equal_variance_test(normal1, normal2, normal = True, alpha = alpha)
        assert result.equal_variance == True
        assert result.pvalue > alpha

    def test_unequal_variance(self, args_normal):
        alpha = 0.05
        normal1, _ , normal3 = args_normal
        result = equal_variance_test(normal1, normal3, normal = True, alpha = alpha)
        assert result.equal_variance == False
        assert result.pvalue <= alpha