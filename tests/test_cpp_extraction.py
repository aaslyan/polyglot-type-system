import unittest
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extractors.cpp_extractor import CppTypeExtractor
from src.types.polyglot_types import *

class TestCppExtraction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize the extractor once for all tests
        cls.extractor = CppTypeExtractor()
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def test_primitive_types(self):
        code = """
        int global_int;
        const double PI = 3.14159;
        unsigned long long big_number;
        """

        cpp_file = Path(self.temp_dir) / "test.cpp"
        cpp_file.write_text(code)

        types = self.extractor.extract_from_file(str(cpp_file))

        # We should find function types for the implicit constructors
        # but the primitive types themselves are built-in

    def test_class_extraction(self):
        code = """
        class TestClass {
        public:
            int public_member;
            void public_method();

        private:
            double private_member;
            void private_method() const;
        };
        """

        cpp_file = Path(self.temp_dir) / "test.cpp"
        cpp_file.write_text(code)

        types = self.extractor.extract_from_file(str(cpp_file))

        # Find TestClass
        test_class = None
        for name, type_obj in types.items():
            if name == "TestClass" and isinstance(type_obj, ObjectType):
                test_class = type_obj
                break

        self.assertIsNotNone(test_class)
        self.assertIn("public_member", test_class.members)
        self.assertIn("public_method", test_class.methods)

    def test_template_extraction(self):
        code = """
        #include <cstddef>
        
        template<typename T>
        class Container {
            T* data;
            size_t size;
        public:
            T& operator[](size_t index);
        };
        """

        cpp_file = Path(self.temp_dir) / "test.cpp"
        cpp_file.write_text(code)

        types = self.extractor.extract_from_file(str(cpp_file))

        # Find Container template
        container = None
        for name, type_obj in types.items():
            if name == "Container" and isinstance(type_obj, TemplateType):
                container = type_obj
                break

        self.assertIsNotNone(container)

if __name__ == "__main__":
    unittest.main()
