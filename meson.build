project('pandas', 'c', 'cpp',
  version: '2.1.4',
  default_options: ['buildtype=release', 'b_ndebug=if-release']
)

py3 = import('python').find_installation('python3')

numpy_dep = py3.get_pkgconfig_variable('numpy', 'pkg-config', required: true)
cython_dep = dependency('cython', method: 'pkg-config', required: true)

sources = files('pandas/_libs/tslibs/base.pyx', 'pandas/_libs/tslibs/dtypes.pyx', 'pandas/_libs/tslibs/ccalendar.pyx')

executable('pandas_ts', sources, dependencies: [cython_dep, numpy_dep])
