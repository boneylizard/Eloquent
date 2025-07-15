import os
import sys
from setuptools import setup, Extension

# --- Get paths and environment variables (same as before) ---
script_dir = os.path.dirname(os.path.realpath(__file__))
llama_cpp_dir = os.path.join(script_dir, "vendor", "llama.cpp")

# --- CUDA Setup ---
cuda_path = os.environ.get('CUDA_PATH')
cuda_include = os.path.join(cuda_path, "include") if cuda_path else None
cuda_lib = os.path.join(cuda_path, "lib", "x64") if cuda_path else None

include_dirs = [
    '.',
    llama_cpp_dir,
    'vendor/llama.cpp/ggml/src',
]
if cuda_include:
     include_dirs.append(cuda_include) #Add cuda include

library_dirs = []
if cuda_lib:
  library_dirs.append(cuda_lib) #Add cuda library

libraries = []
if os.name == 'nt': #Check OS
    libraries.append('kernel32')
    libraries.append('user32')
    libraries.append('gdi32')
    libraries.append('winspool')
    libraries.append('shell32')
    libraries.append('ole32')
    libraries.append('oleaut32')
    libraries.append('uuid')
    libraries.append('comdlg32')
    libraries.append('advapi32')



# --- Compiler Flags ---
# These flags are *crucial* for ensuring compatibility and GPU support.
extra_compile_args = [
    "-mavx2",  # Enable AVX2 instructions (modern CPUs)
    "-mf16c",  # Enable 16-bit floating point conversion
    "-mfma",   # Enable Fused Multiply-Add instructions
    "-Ofast",  # Enable aggressive optimizations
    "-pthread",#Use threads
    "-DLLAMA_CUDA=on" # Make sure this is defined.
]

if os.name != 'nt': #If not windows
  extra_compile_args.append("-Wno-unknown-warning-option")
  extra_compile_args.append("-fpermissive")

# --- Extension Module ---
ext_modules = [
    Extension(
        'llama_cpp',  # This name MUST match what's imported in the Python code
        sources=[
            'llama-cpp-python/llama_cpp.cpp',  # Main C++ source file
            'vendor/llama.cpp/ggml/src/ggml.c',
            'vendor/llama.cpp/ggml/src/ggml-alloc.c',
            'vendor/llama.cpp/ggml/src/ggml-backend.c',
            'vendor/llama.cpp/ggml/src/ggml-quants.c',
            'vendor/llama.cpp/ggml/src/ggml-alloc.c',
            'vendor/llama.cpp/ggml/src/ggml-backend.c',
          #  'vendor/llama.cpp/ggml/src/ggml-metal.m', #Commented out because we want cuda
            'vendor/llama.cpp/ggml/src/ggml-opencl.cpp',
            'vendor/llama.cpp/ggml/src/ggml-backend-mmap.cpp',
            'vendor/llama.cpp/ggml/src/ggml-backend-scratch.cpp',
            'vendor/llama.cpp/ggml/src/ggml-cuda/ggml-cuda.cu',
        ] + (
            [
                "vendor/llama.cpp/ggml/src/ggml-metal.m",
            ]
            if sys.platform == "darwin"
            else []
        ),
        include_dirs=include_dirs,
        library_dirs=library_dirs, #Add Library Dirs
        libraries=libraries,       #Add libraries
        language='c++',
        extra_compile_args=extra_compile_args,
        extra_link_args=[] if os.name == 'nt' else ['-Wl,-rpath,$ORIGIN/.'], #Fix for linux
        define_macros=[
            ("GGML_USE_CUBLAS", "1"),
            ("GGML_CUDA_PEER_MAX_BATCH_SIZE", "128"),
           # ("_XOPEN_SOURCE", "600"), #Add any macros here
        ],
        # --- Other Options ---
        extra_objects=[
            f"{llama_cpp_dir}/ggml/build/Release/ggml-cuda.o",  # Explicitly link CUDA object
        ],

    ),
]

setup(
    name='llama_cpp',
    version='0.0.1',  # Placeholder version - doesn't matter for local build
    ext_modules=ext_modules,
)