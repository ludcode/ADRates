"""
Cavour - Algorithmic Differentiation for Interest Rate Derivatives

Configuration and initialization for the Cavour library.
"""

import os
import jax

# Enable JAX persistent compilation cache for faster subsequent runs
# This reduces JIT warmup time from ~47s to ~5s on 2nd+ executions
_cache_dir = os.path.join(os.path.expanduser('~'), '.cache', 'cavour_jax')
os.makedirs(_cache_dir, exist_ok=True)

jax.config.update('jax_enable_compilation_cache', True)
jax.config.update('jax_compilation_cache_dir', _cache_dir)
jax.config.update('jax_persistent_cache_min_compile_time_secs', 1)

__version__ = '0.1.0'
