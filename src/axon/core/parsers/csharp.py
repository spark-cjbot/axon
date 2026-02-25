"""C# language parser using tree-sitter.

Extracts classes, structs, interfaces, methods, constructors, enums,
imports (using directives), call expressions, type annotation references,
and inheritance relationships from C# source code.
"""

from __future__ import annotations

import tree_sitter_c_sharp as tscsharp
from tree_sitter import Language, Node, Parser

from axon.core.parsers.base import (
    CallInfo,
    ImportInfo,
    LanguageParser,
    ParseResult,
    SymbolInfo,
    TypeRef,
)

CS_LANGUAGE = Language(tscsharp.language())

_BUILTIN_TYPES: frozenset[str] = frozenset(
    {
        "bool",
        "byte",
        "sbyte",
        "char",
        "decimal",
        "double",
        "float",
        "int",
        "uint",
        "long",
        "ulong",
        "short",
        "ushort",
        "string",
        "object",
        "void",
        "dynamic",
        "var",
        "nint",
        "nuint",
    }
)


class CSharpParser(LanguageParser):
    """Parses C# source code using tree-sitter."""

    def __init__(self) -> None:
        self._parser = Parser(CS_LANGUAGE)

    def parse(self, content: str, file_path: str) -> ParseResult:
        """Parse C# source and return structured information."""
        tree = self._parser.parse(bytes(content, "utf8"))
        result = ParseResult()
        self._walk(tree.root_node, content, result, class_name="")
        return result

    # ------------------------------------------------------------------
    # AST walking
    # ------------------------------------------------------------------

    def _walk(
        self,
        node: Node,
        content: str,
        result: ParseResult,
        class_name: str,
    ) -> None:
        """Recursively walk the AST to extract definitions and calls."""
        for child in node.children:
            ntype = child.type
            if ntype == "using_directive":
                self._extract_using(child, result)
            elif ntype == "namespace_declaration":
                self._extract_namespace(child, content, result, class_name)
            elif ntype == "file_scoped_namespace_declaration":
                self._extract_namespace(child, content, result, class_name)
            elif ntype == "class_declaration":
                self._extract_class(child, content, result)
            elif ntype == "struct_declaration":
                self._extract_struct(child, content, result)
            elif ntype == "interface_declaration":
                self._extract_interface(child, content, result)
            elif ntype == "enum_declaration":
                self._extract_enum(child, content, result)
            elif ntype == "record_declaration":
                self._extract_class(child, content, result)
            elif ntype == "method_declaration":
                self._extract_method(child, content, result, class_name)
            elif ntype == "constructor_declaration":
                self._extract_constructor(child, content, result, class_name)
            elif ntype == "invocation_expression":
                self._extract_call(child, result)
            elif ntype == "object_creation_expression":
                self._extract_new_expression(child, result)
            elif ntype == "expression_statement":
                self._walk_expression_for_calls(child, result)
            elif ntype == "local_declaration_statement":
                self._extract_local_variable_types(child, result)
                self._walk_expression_for_calls(child, result)
            elif ntype == "return_statement":
                self._walk_expression_for_calls(child, result)
            elif ntype in ("block", "declaration_list"):
                self._walk(child, content, result, class_name)
            elif ntype == "global_statement":
                self._walk(child, content, result, class_name)
            else:
                # Recurse into unknown containers
                if child.child_count > 0 and ntype not in (
                    "modifier",
                    "attribute_list",
                    "parameter_list",
                    "type_parameter_list",
                    "base_list",
                    "argument_list",
                ):
                    pass  # don't recurse into leaf-like nodes

    # ------------------------------------------------------------------
    # Using directives (imports)
    # ------------------------------------------------------------------

    def _extract_using(self, node: Node, result: ParseResult) -> None:
        """Extract a ``using`` directive as an import."""
        for child in node.children:
            if child.type in ("identifier", "qualified_name"):
                module = child.text.decode("utf8")
                parts = module.split(".")
                result.imports.append(
                    ImportInfo(
                        module=module,
                        names=[parts[-1]],
                    )
                )
                return

    # ------------------------------------------------------------------
    # Namespaces
    # ------------------------------------------------------------------

    def _extract_namespace(
        self,
        node: Node,
        content: str,
        result: ParseResult,
        class_name: str,
    ) -> None:
        """Walk into a namespace declaration to find type definitions."""
        for child in node.children:
            if child.type in ("declaration_list",):
                self._walk(child, content, result, class_name)
            elif child.type in (
                "class_declaration",
                "struct_declaration",
                "interface_declaration",
                "enum_declaration",
                "record_declaration",
            ):
                # file-scoped namespaces put declarations as direct children
                self._walk_single(child, content, result, class_name)

    def _walk_single(
        self,
        child: Node,
        content: str,
        result: ParseResult,
        class_name: str,
    ) -> None:
        """Process a single declaration node."""
        ntype = child.type
        if ntype == "class_declaration":
            self._extract_class(child, content, result)
        elif ntype == "struct_declaration":
            self._extract_struct(child, content, result)
        elif ntype == "interface_declaration":
            self._extract_interface(child, content, result)
        elif ntype == "enum_declaration":
            self._extract_enum(child, content, result)
        elif ntype == "record_declaration":
            self._extract_class(child, content, result)

    # ------------------------------------------------------------------
    # Classes and structs
    # ------------------------------------------------------------------

    def _extract_class(
        self,
        node: Node,
        content: str,
        result: ParseResult,
    ) -> None:
        """Extract a class or record declaration."""
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return

        class_name = name_node.text.decode("utf8")
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        node_content = content[node.start_byte : node.end_byte]

        decorators = self._extract_attributes(node)

        result.symbols.append(
            SymbolInfo(
                name=class_name,
                kind="class",
                start_line=start_line,
                end_line=end_line,
                content=node_content,
                decorators=decorators,
            )
        )

        self._extract_base_list(node, class_name, result)
        self._extract_exports(node, class_name, result)

        # Walk class body for methods, constructors, nested types
        body = node.child_by_field_name("body")
        if body is None:
            for child in node.children:
                if child.type == "declaration_list":
                    body = child
                    break

        if body is not None:
            self._walk(body, content, result, class_name=class_name)

    def _extract_struct(
        self,
        node: Node,
        content: str,
        result: ParseResult,
    ) -> None:
        """Extract a struct declaration (treated as a class)."""
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return

        struct_name = name_node.text.decode("utf8")
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        node_content = content[node.start_byte : node.end_byte]

        result.symbols.append(
            SymbolInfo(
                name=struct_name,
                kind="class",
                start_line=start_line,
                end_line=end_line,
                content=node_content,
            )
        )

        self._extract_base_list(node, struct_name, result)
        self._extract_exports(node, struct_name, result)

        body = node.child_by_field_name("body")
        if body is None:
            for child in node.children:
                if child.type == "declaration_list":
                    body = child
                    break

        if body is not None:
            self._walk(body, content, result, class_name=struct_name)

    # ------------------------------------------------------------------
    # Interfaces
    # ------------------------------------------------------------------

    def _extract_interface(
        self,
        node: Node,
        content: str,
        result: ParseResult,
    ) -> None:
        """Extract an interface declaration."""
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return

        name = name_node.text.decode("utf8")
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        node_content = content[node.start_byte : node.end_byte]

        result.symbols.append(
            SymbolInfo(
                name=name,
                kind="interface",
                start_line=start_line,
                end_line=end_line,
                content=node_content,
            )
        )

        # Interface base list: all are "extends" (interface inheritance)
        for child in node.children:
            if child.type == "base_list":
                for sub in child.children:
                    if sub.type in ("identifier", "generic_name"):
                        parent_name = self._type_name(sub)
                        if parent_name:
                            result.heritage.append((name, "extends", parent_name))

        self._extract_exports(node, name, result)

        body = node.child_by_field_name("body")
        if body is None:
            for child in node.children:
                if child.type == "declaration_list":
                    body = child
                    break

        if body is not None:
            self._walk(body, content, result, class_name=name)

    # ------------------------------------------------------------------
    # Enums
    # ------------------------------------------------------------------

    def _extract_enum(
        self,
        node: Node,
        content: str,
        result: ParseResult,
    ) -> None:
        """Extract an enum declaration."""
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return

        name = name_node.text.decode("utf8")
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        node_content = content[node.start_byte : node.end_byte]

        result.symbols.append(
            SymbolInfo(
                name=name,
                kind="enum",
                start_line=start_line,
                end_line=end_line,
                content=node_content,
            )
        )

    # ------------------------------------------------------------------
    # Methods and constructors
    # ------------------------------------------------------------------

    def _extract_method(
        self,
        node: Node,
        content: str,
        result: ParseResult,
        class_name: str,
    ) -> None:
        """Extract a method declaration."""
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return

        name = name_node.text.decode("utf8")
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        node_content = content[node.start_byte : node.end_byte]

        kind = "method" if class_name else "function"
        signature = self._build_method_signature(node, name)
        decorators = self._extract_attributes(node)

        result.symbols.append(
            SymbolInfo(
                name=name,
                kind=kind,
                start_line=start_line,
                end_line=end_line,
                content=node_content,
                signature=signature,
                class_name=class_name,
                decorators=decorators,
            )
        )

        # Extract parameter types
        self._extract_param_types(node, result)

        # Extract return type
        return_type = self._get_return_type(node)
        if return_type and return_type not in _BUILTIN_TYPES:
            result.type_refs.append(
                TypeRef(
                    name=return_type,
                    kind="return",
                    line=start_line,
                )
            )

        # Walk method body for calls
        for child in node.children:
            if child.type == "block":
                self._walk(child, content, result, class_name=class_name)

    def _extract_constructor(
        self,
        node: Node,
        content: str,
        result: ParseResult,
        class_name: str,
    ) -> None:
        """Extract a constructor declaration."""
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return

        name = name_node.text.decode("utf8")
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        node_content = content[node.start_byte : node.end_byte]

        # Constructors are named after the class — use ".ctor" convention
        signature = self._build_constructor_signature(node, name)

        result.symbols.append(
            SymbolInfo(
                name=name,
                kind="method",
                start_line=start_line,
                end_line=end_line,
                content=node_content,
                signature=signature,
                class_name=class_name,
            )
        )

        self._extract_param_types(node, result)

        # Walk constructor body for calls
        for child in node.children:
            if child.type == "block":
                self._walk(child, content, result, class_name=class_name)

    # ------------------------------------------------------------------
    # Heritage (base classes / interfaces)
    # ------------------------------------------------------------------

    def _extract_base_list(
        self,
        node: Node,
        class_name: str,
        result: ParseResult,
    ) -> None:
        """Extract inheritance from a base_list child.

        Uses the C# convention: names starting with ``I`` followed by an
        uppercase letter are treated as interfaces (``implements``), all
        others are base classes (``extends``).
        """
        for child in node.children:
            if child.type != "base_list":
                continue
            for sub in child.children:
                if sub.type in ("identifier", "generic_name"):
                    parent_name = self._type_name(sub)
                    if not parent_name:
                        continue
                    kind = self._heritage_kind(parent_name)
                    result.heritage.append((class_name, kind, parent_name))

    @staticmethod
    def _heritage_kind(name: str) -> str:
        """Guess whether *name* is a class (extends) or interface (implements)."""
        if len(name) >= 2 and name[0] == "I" and name[1].isupper():
            return "implements"
        return "extends"

    # ------------------------------------------------------------------
    # Call extraction
    # ------------------------------------------------------------------

    def _extract_call(self, node: Node, result: ParseResult) -> None:
        """Extract an invocation_expression."""
        line = node.start_point[0] + 1
        arguments = self._extract_identifier_arguments(node)

        # The function being called is the first child
        func_node = node.children[0] if node.children else None
        if func_node is None:
            return

        if func_node.type == "member_access_expression":
            name, receiver = self._extract_member_access(func_node)
            result.calls.append(
                CallInfo(
                    name=name,
                    line=line,
                    receiver=receiver,
                    arguments=arguments,
                )
            )
        elif func_node.type == "identifier":
            result.calls.append(
                CallInfo(
                    name=func_node.text.decode("utf8"),
                    line=line,
                    arguments=arguments,
                )
            )

    def _extract_new_expression(self, node: Node, result: ParseResult) -> None:
        """Extract an object_creation_expression (``new Type(args)``)."""
        line = node.start_point[0] + 1
        arguments = self._extract_identifier_arguments(node)

        for child in node.children:
            if child.type in ("identifier", "generic_name"):
                type_name = self._type_name(child)
                if type_name:
                    result.calls.append(
                        CallInfo(
                            name=type_name,
                            line=line,
                            arguments=arguments,
                        )
                    )
                return
            if child.type == "qualified_name":
                # new Namespace.Type(args) — use the last identifier
                type_name = self._last_identifier(child)
                if type_name:
                    result.calls.append(
                        CallInfo(
                            name=type_name,
                            line=line,
                            arguments=arguments,
                        )
                    )
                return

    def _walk_expression_for_calls(self, node: Node, result: ParseResult) -> None:
        """Recursively find invocation and object_creation expressions."""
        for child in node.children:
            if child.type == "invocation_expression":
                self._extract_call(child, result)
            elif child.type == "object_creation_expression":
                self._extract_new_expression(child, result)
            if child.child_count > 0:
                self._walk_expression_for_calls(child, result)

    @staticmethod
    def _extract_member_access(node: Node) -> tuple[str, str]:
        """Extract (method_name, receiver) from a member_access_expression."""
        parts: list[str] = []
        for child in node.children:
            if child.type == "identifier":
                parts.append(child.text.decode("utf8"))

        if len(parts) >= 2:
            return parts[-1], parts[0]
        if len(parts) == 1:
            return parts[0], ""
        return "", ""

    @staticmethod
    def _extract_identifier_arguments(node: Node) -> list[str]:
        """Extract bare identifier arguments from argument_list."""
        for child in node.children:
            if child.type == "argument_list":
                identifiers: list[str] = []
                for arg in child.children:
                    if arg.type == "argument":
                        for sub in arg.children:
                            if sub.type == "identifier":
                                identifiers.append(sub.text.decode("utf8"))
                return identifiers
        return []

    # ------------------------------------------------------------------
    # Type references
    # ------------------------------------------------------------------

    def _extract_param_types(self, node: Node, result: ParseResult) -> None:
        """Extract type annotations from method/constructor parameters."""
        for child in node.children:
            if child.type == "parameter_list":
                for param in child.children:
                    if param.type == "parameter":
                        self._extract_single_param_type(param, result)

    def _extract_single_param_type(self, param: Node, result: ParseResult) -> None:
        """Extract type from a single parameter node."""
        param_name = ""
        type_name = ""

        for child in param.children:
            if child.type == "identifier":
                param_name = child.text.decode("utf8")
            elif child.type in (
                "predefined_type",
                "generic_name",
                "nullable_type",
                "array_type",
            ):
                type_name = self._type_name(child)
            elif child.type == "identifier" and not param_name:
                # First identifier could be a type if it's a user-defined type
                pass

        # In C# parameters, type comes before name: "User user"
        # Both are identifier nodes, so we need to handle this
        identifiers = [c for c in param.children if c.type == "identifier"]
        if len(identifiers) >= 2 and not type_name:
            type_name = identifiers[0].text.decode("utf8")
            param_name = identifiers[1].text.decode("utf8")
        elif len(identifiers) == 1 and not type_name:
            # Single identifier — it's the name, type is a predefined type
            param_name = identifiers[0].text.decode("utf8")

        if type_name and type_name not in _BUILTIN_TYPES:
            result.type_refs.append(
                TypeRef(
                    name=type_name,
                    kind="param",
                    line=param.start_point[0] + 1,
                    param_name=param_name,
                )
            )

    def _extract_local_variable_types(self, node: Node, result: ParseResult) -> None:
        """Extract type references from local variable declarations."""
        for child in node.children:
            if child.type == "variable_declaration":
                for sub in child.children:
                    if sub.type in ("identifier", "generic_name"):
                        type_name = self._type_name(sub)
                        if type_name and type_name not in _BUILTIN_TYPES:
                            result.type_refs.append(
                                TypeRef(
                                    name=type_name,
                                    kind="variable",
                                    line=sub.start_point[0] + 1,
                                )
                            )
                        return
                    elif sub.type in ("predefined_type", "implicit_type"):
                        return  # built-in or var, skip

    def _get_return_type(self, method_node: Node) -> str:
        """Get the return type name from a method declaration.

        In C# the return type appears as a child before the method name.
        It can be a ``predefined_type``, ``identifier``, or ``generic_name``.
        """
        for child in method_node.children:
            if child.type == "predefined_type":
                return child.text.decode("utf8")
            if child.type in ("identifier", "generic_name"):
                # Check if this is the method name or the return type
                # by seeing if the next sibling is the method name
                name_node = method_node.child_by_field_name("name")
                if name_node is not None and child.id != name_node.id:
                    return self._type_name(child)
            if child.type in ("nullable_type", "array_type"):
                return self._type_name(child)
        return ""

    # ------------------------------------------------------------------
    # Attributes (C# decorators)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_attributes(node: Node) -> list[str]:
        """Extract attribute names from ``attribute_list`` children.

        Maps C# attributes (``[HttpGet]``, ``[Route("/api")]``) to
        decorator-like names for consistency with the base schema.
        """
        attrs: list[str] = []
        for child in node.children:
            if child.type == "attribute_list":
                for sub in child.children:
                    if sub.type == "attribute":
                        name_node = sub.child_by_field_name("name")
                        if name_node is None:
                            # Fallback: first identifier child
                            for attr_child in sub.children:
                                if attr_child.type == "identifier":
                                    name_node = attr_child
                                    break
                        if name_node is not None:
                            attrs.append(name_node.text.decode("utf8"))
        return attrs

    # ------------------------------------------------------------------
    # Exports
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_exports(node: Node, name: str, result: ParseResult) -> None:
        """Mark public types as exported."""
        for child in node.children:
            if child.type == "modifier":
                for sub in child.children:
                    if sub.type == "public":
                        result.exports.append(name)
                        return

    # ------------------------------------------------------------------
    # Signature building
    # ------------------------------------------------------------------

    def _build_method_signature(self, node: Node, name: str) -> str:
        """Build a human-readable signature for a method."""
        return_type = self._get_return_type(node)
        params = ""
        for child in node.children:
            if child.type == "parameter_list":
                params = child.text.decode("utf8")
                break

        sig = f"{name}{params}"
        if return_type:
            sig = f"{return_type} {sig}"
        return sig

    @staticmethod
    def _build_constructor_signature(node: Node, name: str) -> str:
        """Build a human-readable signature for a constructor."""
        params = ""
        for child in node.children:
            if child.type == "parameter_list":
                params = child.text.decode("utf8")
                break
        return f"{name}{params}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _type_name(node: Node) -> str:
        """Extract the primary type name from a type node.

        For ``identifier`` returns the text directly.
        For ``generic_name`` like ``List<User>`` returns ``List``.
        For ``nullable_type`` like ``User?`` returns ``User``.
        For ``array_type`` like ``User[]`` returns ``User``.
        """
        if node.type == "identifier":
            return node.text.decode("utf8")
        if node.type == "generic_name":
            for child in node.children:
                if child.type == "identifier":
                    return child.text.decode("utf8")
        if node.type in ("nullable_type", "array_type"):
            for child in node.children:
                if child.type == "identifier":
                    return child.text.decode("utf8")
                if child.type == "generic_name":
                    for sub in child.children:
                        if sub.type == "identifier":
                            return sub.text.decode("utf8")
                if child.type == "predefined_type":
                    return child.text.decode("utf8")
        if node.type == "predefined_type":
            return node.text.decode("utf8")
        return ""

    @staticmethod
    def _last_identifier(node: Node) -> str:
        """Return the text of the last identifier child."""
        last = ""
        for child in node.children:
            if child.type == "identifier":
                last = child.text.decode("utf8")
        return last
