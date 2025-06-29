import clang.cindex
import platform
import os
from typing import Optional

def setup_clang(lib_path: Optional[str] = None):
    """Setup clang library path"""
    if lib_path:
        clang.cindex.Config.set_library_file(lib_path)
    else:
        # Try to find clang library automatically
        system = platform.system()

        if system == "Linux":
            possible_paths = [
                "/usr/lib/llvm-14/lib/libclang.so.1",
                "/usr/lib/llvm-13/lib/libclang.so.1",
                "/usr/lib/llvm-12/lib/libclang.so.1",
                "/usr/lib/x86_64-linux-gnu/libclang-14.so.1",
                "/usr/lib/libclang.so",
            ]
        elif system == "Darwin":  # macOS
            possible_paths = [
                "/Library/Developer/CommandLineTools/usr/lib/libclang.dylib",
                "/usr/local/opt/llvm/lib/libclang.dylib",
                "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/lib/libclang.dylib",
            ]
        elif system == "Windows":
            possible_paths = [
                r"C:\Program Files\LLVM\bin\libclang.dll",
                r"C:\Program Files (x86)\LLVM\bin\libclang.dll",
            ]
        else:
            possible_paths = []

        for path in possible_paths:
            if os.path.exists(path):
                clang.cindex.Config.set_library_file(path)
                break

def get_type_spelling(clang_type: clang.cindex.Type) -> str:
    """Get clean type spelling"""
    spelling = clang_type.spelling

    # Clean up some common patterns
    spelling = spelling.replace("struct ", "")
    spelling = spelling.replace("class ", "")
    spelling = spelling.replace("enum ", "")

    return spelling
