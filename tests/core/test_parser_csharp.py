"""Tests for the C# language parser."""

from __future__ import annotations

import pytest

from axon.core.parsers.csharp import CSharpParser


@pytest.fixture
def parser() -> CSharpParser:
    return CSharpParser()


# ---------------------------------------------------------------------------
# Classes
# ---------------------------------------------------------------------------


class TestParseClass:
    """Parse a class with methods."""

    CODE = (
        "public class User\n"
        "{\n"
        "    public string Name { get; set; }\n"
        "\n"
        "    public User(string name)\n"
        "    {\n"
        "        Name = name;\n"
        "    }\n"
        "\n"
        "    public bool Save()\n"
        "    {\n"
        "        return true;\n"
        "    }\n"
        "}\n"
    )

    def test_symbol_count(self, parser: CSharpParser) -> None:
        result = parser.parse(self.CODE, "User.cs")
        # 1 class + 1 constructor + 1 method = 3
        assert len(result.symbols) == 3

    def test_class_symbol(self, parser: CSharpParser) -> None:
        result = parser.parse(self.CODE, "User.cs")
        cls = [s for s in result.symbols if s.kind == "class"]
        assert len(cls) == 1
        assert cls[0].name == "User"

    def test_method_count(self, parser: CSharpParser) -> None:
        result = parser.parse(self.CODE, "User.cs")
        # Constructor is now kind="constructor", not "method" — only Save remains.
        methods = [s for s in result.symbols if s.kind == "method"]
        assert len(methods) == 1

    def test_method_class_name(self, parser: CSharpParser) -> None:
        result = parser.parse(self.CODE, "User.cs")
        methods = [s for s in result.symbols if s.kind == "method"]
        for method in methods:
            assert method.class_name == "User"

    def test_method_names(self, parser: CSharpParser) -> None:
        result = parser.parse(self.CODE, "User.cs")
        # Constructor is now stored as kind="constructor" with name=".ctor"
        method_names = {s.name for s in result.symbols if s.kind == "method"}
        assert method_names == {"Save"}

    def test_constructor_kind(self, parser: CSharpParser) -> None:
        result = parser.parse(self.CODE, "User.cs")
        ctors = [s for s in result.symbols if s.kind == "constructor"]
        assert len(ctors) == 1
        assert ctors[0].name == ".ctor"
        assert ctors[0].class_name == "User"

    def test_constructor_signature_contains_class_name(self, parser: CSharpParser) -> None:
        result = parser.parse(self.CODE, "User.cs")
        ctors = [s for s in result.symbols if s.kind == "constructor"]
        assert len(ctors) == 1
        assert "User" in ctors[0].signature

    def test_public_class_is_exported(self, parser: CSharpParser) -> None:
        result = parser.parse(self.CODE, "User.cs")
        assert "User" in result.exports


# ---------------------------------------------------------------------------
# Inheritance
# ---------------------------------------------------------------------------


class TestParseInheritance:
    """Parse class inheritance (heritage)."""

    def test_single_base_class(self, parser: CSharpParser) -> None:
        code = "public class Admin : User\n{\n}\n"
        result = parser.parse(code, "Admin.cs")
        assert ("Admin", "extends", "User") in result.heritage

    def test_interface_implementation(self, parser: CSharpParser) -> None:
        code = "public class UserRepo : IRepository\n{\n}\n"
        result = parser.parse(code, "UserRepo.cs")
        assert ("UserRepo", "implements", "IRepository") in result.heritage

    def test_class_and_interfaces(self, parser: CSharpParser) -> None:
        code = "public class Admin : User, ISerializable, IDisposable\n{\n}\n"
        result = parser.parse(code, "Admin.cs")
        assert ("Admin", "extends", "User") in result.heritage
        assert ("Admin", "implements", "ISerializable") in result.heritage
        assert ("Admin", "implements", "IDisposable") in result.heritage

    def test_no_base(self, parser: CSharpParser) -> None:
        code = "public class Plain\n{\n}\n"
        result = parser.parse(code, "Plain.cs")
        assert len(result.heritage) == 0

    def test_interface_extends_interface(self, parser: CSharpParser) -> None:
        code = "public interface IRepo : IDisposable\n{\n}\n"
        result = parser.parse(code, "IRepo.cs")
        assert ("IRepo", "extends", "IDisposable") in result.heritage


# ---------------------------------------------------------------------------
# Imports (using directives)
# ---------------------------------------------------------------------------


class TestParseImports:
    """Parse using directives."""

    CODE = (
        "using System;\n"
        "using System.Collections.Generic;\n"
        "using Newtonsoft.Json;\n"
    )

    def test_import_count(self, parser: CSharpParser) -> None:
        result = parser.parse(self.CODE, "Program.cs")
        assert len(result.imports) == 3

    def test_system_import(self, parser: CSharpParser) -> None:
        result = parser.parse(self.CODE, "Program.cs")
        sys_imp = [i for i in result.imports if i.module == "System"]
        assert len(sys_imp) == 1
        assert "System" in sys_imp[0].names

    def test_qualified_import(self, parser: CSharpParser) -> None:
        result = parser.parse(self.CODE, "Program.cs")
        col_imp = [i for i in result.imports if i.module == "System.Collections.Generic"]
        assert len(col_imp) == 1
        assert "Generic" in col_imp[0].names


# ---------------------------------------------------------------------------
# Function calls
# ---------------------------------------------------------------------------


class TestParseFunctionCalls:
    """Parse invocation and object creation expressions."""

    CODE = (
        "public class Service\n"
        "{\n"
        "    public void Run()\n"
        "    {\n"
        "        var result = repository.FindById(id);\n"
        "        Console.WriteLine(result.Name);\n"
        "        var user = new User(\"test\");\n"
        "        DoSomething();\n"
        "    }\n"
        "}\n"
    )

    def test_method_call(self, parser: CSharpParser) -> None:
        result = parser.parse(self.CODE, "Service.cs")
        find_calls = [c for c in result.calls if c.name == "FindById"]
        assert len(find_calls) == 1
        assert find_calls[0].receiver == "repository"

    def test_static_call(self, parser: CSharpParser) -> None:
        result = parser.parse(self.CODE, "Service.cs")
        wl_calls = [c for c in result.calls if c.name == "WriteLine"]
        assert len(wl_calls) == 1
        assert wl_calls[0].receiver == "Console"

    def test_new_expression(self, parser: CSharpParser) -> None:
        result = parser.parse(self.CODE, "Service.cs")
        new_calls = [c for c in result.calls if c.name == "User"]
        assert len(new_calls) == 1

    def test_simple_call(self, parser: CSharpParser) -> None:
        result = parser.parse(self.CODE, "Service.cs")
        do_calls = [c for c in result.calls if c.name == "DoSomething"]
        assert len(do_calls) == 1
        assert do_calls[0].receiver == ""


# ---------------------------------------------------------------------------
# Type annotations
# ---------------------------------------------------------------------------


class TestParseTypeAnnotations:
    """Parse type annotations from parameters, return types, and variables."""

    CODE = (
        "public class Handler\n"
        "{\n"
        "    public Response Handle(User user, Config config)\n"
        "    {\n"
        "        AuthResult result = Authenticate(user);\n"
        "        return new Response();\n"
        "    }\n"
        "}\n"
    )

    def test_param_types(self, parser: CSharpParser) -> None:
        result = parser.parse(self.CODE, "Handler.cs")
        param_refs = [t for t in result.type_refs if t.kind == "param"]
        param_names = {t.name for t in param_refs}
        assert "User" in param_names
        assert "Config" in param_names

    def test_param_names_attached(self, parser: CSharpParser) -> None:
        result = parser.parse(self.CODE, "Handler.cs")
        user_ref = [t for t in result.type_refs if t.name == "User" and t.kind == "param"]
        assert len(user_ref) == 1
        assert user_ref[0].param_name == "user"

    def test_return_type(self, parser: CSharpParser) -> None:
        result = parser.parse(self.CODE, "Handler.cs")
        return_refs = [t for t in result.type_refs if t.kind == "return"]
        assert any(t.name == "Response" for t in return_refs)

    def test_variable_type(self, parser: CSharpParser) -> None:
        result = parser.parse(self.CODE, "Handler.cs")
        var_refs = [t for t in result.type_refs if t.kind == "variable"]
        assert any(t.name == "AuthResult" for t in var_refs)

    def test_builtin_types_skipped(self, parser: CSharpParser) -> None:
        code = (
            "public class Foo\n"
            "{\n"
            "    public int Add(int x, string y)\n"
            "    {\n"
            "        return 0;\n"
            "    }\n"
            "}\n"
        )
        result = parser.parse(code, "Foo.cs")
        # int and string are built-in — should produce no type_refs
        assert len(result.type_refs) == 0


# ---------------------------------------------------------------------------
# Interfaces
# ---------------------------------------------------------------------------


class TestParseInterface:
    """Parse interface declarations."""

    def test_interface_symbol(self, parser: CSharpParser) -> None:
        code = (
            "public interface IUserService\n"
            "{\n"
            "    User GetById(int id);\n"
            "    void Save(User user);\n"
            "}\n"
        )
        result = parser.parse(code, "IUserService.cs")
        interfaces = [s for s in result.symbols if s.kind == "interface"]
        assert len(interfaces) == 1
        assert interfaces[0].name == "IUserService"

    def test_interface_methods(self, parser: CSharpParser) -> None:
        code = (
            "public interface IUserService\n"
            "{\n"
            "    User GetById(int id);\n"
            "    void Save(User user);\n"
            "}\n"
        )
        result = parser.parse(code, "IUserService.cs")
        methods = [s for s in result.symbols if s.kind == "method"]
        method_names = {m.name for m in methods}
        assert "GetById" in method_names
        assert "Save" in method_names


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TestParseEnum:
    """Parse enum declarations."""

    def test_enum_symbol(self, parser: CSharpParser) -> None:
        code = (
            "public enum UserRole\n"
            "{\n"
            "    Admin,\n"
            "    Member,\n"
            "    Guest\n"
            "}\n"
        )
        result = parser.parse(code, "UserRole.cs")
        enums = [s for s in result.symbols if s.kind == "enum"]
        assert len(enums) == 1
        assert enums[0].name == "UserRole"


# ---------------------------------------------------------------------------
# Namespaces
# ---------------------------------------------------------------------------


class TestParseNamespace:
    """Parse types inside namespace declarations."""

    def test_class_inside_namespace(self, parser: CSharpParser) -> None:
        code = (
            "namespace MyApp.Models\n"
            "{\n"
            "    public class User\n"
            "    {\n"
            "    }\n"
            "}\n"
        )
        result = parser.parse(code, "User.cs")
        classes = [s for s in result.symbols if s.kind == "class"]
        assert len(classes) == 1
        assert classes[0].name == "User"


# ---------------------------------------------------------------------------
# Attributes (decorators)
# ---------------------------------------------------------------------------


class TestParseAttributes:
    """C# attributes are captured as decorators."""

    def test_single_attribute(self, parser: CSharpParser) -> None:
        code = (
            "public class Controller\n"
            "{\n"
            "    [HttpGet]\n"
            "    public void GetAll()\n"
            "    {\n"
            "    }\n"
            "}\n"
        )
        result = parser.parse(code, "Controller.cs")
        methods = [s for s in result.symbols if s.kind == "method"]
        assert len(methods) == 1
        assert "HttpGet" in methods[0].decorators

    def test_multiple_attributes(self, parser: CSharpParser) -> None:
        code = (
            "public class Controller\n"
            "{\n"
            "    [HttpGet]\n"
            '    [Route("/api/users")]\n'
            "    public void GetAll()\n"
            "    {\n"
            "    }\n"
            "}\n"
        )
        result = parser.parse(code, "Controller.cs")
        methods = [s for s in result.symbols if s.kind == "method"]
        assert len(methods) == 1
        assert "HttpGet" in methods[0].decorators
        assert "Route" in methods[0].decorators

    def test_class_attribute(self, parser: CSharpParser) -> None:
        code = (
            "[ApiController]\n"
            "public class UsersController\n"
            "{\n"
            "}\n"
        )
        result = parser.parse(code, "UsersController.cs")
        classes = [s for s in result.symbols if s.kind == "class"]
        assert len(classes) == 1
        assert "ApiController" in classes[0].decorators


# ---------------------------------------------------------------------------
# Struct
# ---------------------------------------------------------------------------


class TestParseStruct:
    """Parse struct declarations (treated as class kind)."""

    def test_struct_symbol(self, parser: CSharpParser) -> None:
        code = (
            "public struct Point\n"
            "{\n"
            "    public int X { get; set; }\n"
            "    public int Y { get; set; }\n"
            "}\n"
        )
        result = parser.parse(code, "Point.cs")
        classes = [s for s in result.symbols if s.kind == "class"]
        assert len(classes) == 1
        assert classes[0].name == "Point"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases and less common patterns."""

    def test_empty_file(self, parser: CSharpParser) -> None:
        result = parser.parse("", "empty.cs")
        assert result.symbols == []
        assert result.imports == []
        assert result.calls == []
        assert result.type_refs == []
        assert result.heritage == []

    def test_syntax_error_does_not_crash(self, parser: CSharpParser) -> None:
        code = "public class Broken {\n"
        result = parser.parse(code, "broken.cs")
        assert isinstance(result, type(result))

    def test_method_signature(self, parser: CSharpParser) -> None:
        code = (
            "public class Svc\n"
            "{\n"
            "    public string GetName(int id)\n"
            "    {\n"
            "        return \"\";\n"
            "    }\n"
            "}\n"
        )
        result = parser.parse(code, "Svc.cs")
        methods = [s for s in result.symbols if s.name == "GetName"]
        assert len(methods) == 1
        assert "GetName" in methods[0].signature
        assert "(int id)" in methods[0].signature
