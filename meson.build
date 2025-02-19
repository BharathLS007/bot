project('pandas', 'c', 'cpp',
  version: '2.1.4',
  default_options: ['buildtype=release', 'b_ndebug=if-release']
)

py3 = import('python').find_installation('python3')

sources = files('src/pandas/_libs/tslibs/base.pyx', 'src/pandas/_libs/tslibs/dtypes.pyx', 'src/pandas/_libs/tslibs/ccalendar.pyx')

executable('pandas_ts', sources)
