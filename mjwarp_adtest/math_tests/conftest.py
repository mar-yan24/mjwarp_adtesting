"""Math test-specific fixtures and configuration."""

import pytest


@pytest.fixture
def fd_eps(ad_config):
  return ad_config.fd_eps


@pytest.fixture
def fd_tol(ad_config):
  return ad_config.fd_tol


@pytest.fixture
def contact_fd_tol(ad_config):
  return ad_config.contact_fd_tol
