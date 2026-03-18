"""Shared test infrastructure: data factories, loss functions, FD utilities."""

from mjwarp_adtest.fixtures.data_factory import make_ad_fixture
from mjwarp_adtest.fixtures.data_factory import make_baseline_fixture
from mjwarp_adtest.fixtures.finite_difference import fd_gradient
from mjwarp_adtest.fixtures.finite_difference import fd_jacobian
from mjwarp_adtest.fixtures.finite_difference import taylor_test
