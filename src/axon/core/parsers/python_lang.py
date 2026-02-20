"""Python language parser using tree-sitter.

Extracts functions, classes, methods, imports, calls, type annotations,
and inheritance relationships from Python source code.
"""

from __future__ import annotations

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

from axon.core.parsers.base import (
    CallInfo,
    ImportInfo,
    LanguageParser,
    ParseResult,
    SymbolInfo,
    TypeRef,
)

PY_LANGUAGE = Language(tspython.language())

# Built-in types to skip when extracting type references.
_BUILTIN_TYPES: frozenset[str] = frozenset(
    {
        "str",
        "int",
        "float",
        "bool",
        "None",
        "list",
        "dict",
        "set",
        "tuple",
        "Any",
        "Optional",
        "bytes",
        "complex",
        "object",
        "type",
    }
)


class PythonParser(LanguageParser):
    """Parses Python source code using tree-sitter."""

    def __init__(self) -> None:
        self._parser = Parser(PY_LANGUAGE)

    def parse(self, content: str, file_path: str) -> ParseResult:
        """Parse Python source and return structured information."""
        tree = self._parser.parse(bytes(content, "utf8"))
        result = ParseResult()
        root = tree.root_node
        self._walk(root, content, result, class_name="")
        # Extract module-level calls (e.g. ``setup()`` at the top of a script).
        self._extract_calls_recursive(root, result)
        return result

    # ------------------------------------------------------------------
    # Tree walking
    # ------------------------------------------------------------------

    def _walk(
        self,
        node: Node,
        content: str,
        result: ParseResult,
        class_name: str,
    ) -> None:
        """Recursively walk the AST to extract definitions and annotations.

        Call extraction is handled separately via ``_extract_calls_recursive``
        at each scope boundary (module, class, function) to avoid
        double-counting.
        """
        for child in node.children:
            match child.type:
                case "function_definition":
                    self._extract_function(child, content, result, class_name)
                case "class_definition":
                    self._extract_class(child, content, result)
                case "import_statement":
                    self._extract_import(child, result)
                case "import_from_statement":
                    self._extract_import_from(child, result)
                case "expression_statement":
                    # Only extract variable annotations here; calls are
                    # handled by the scope-level _extract_calls_recursive.
                    self._extract_annotations_from_expression(child, result)
                case _:
                    self._walk(child, content, result, class_name)

    # ------------------------------------------------------------------
    # Functions and methods
    # ------------------------------------------------------------------

    def _extract_function(
        self,
        node: Node,
        content: str,
        result: ParseResult,
        class_name: str,
    ) -> None:
        """Extract a function or method definition."""
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return

        name = name_node.text.decode("utf8")
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        node_content = content[node.start_byte : node.end_byte]

        kind = "method" if class_name else "function"
        signature = self._build_signature(node, content)

        result.symbols.append(
            SymbolInfo(
                name=name,
                kind=kind,
                start_line=start_line,
                end_line=end_line,
                content=node_content,
                signature=signature,
                class_name=class_name,
            )
        )

        # Extract parameter type annotations.
        self._extract_param_types(node, result)

        # Extract return type annotation.
        return_type = node.child_by_field_name("return_type")
        if return_type is not None:
            type_name = self._extract_type_name(return_type)
            if type_name and type_name not in _BUILTIN_TYPES:
                result.type_refs.append(
                    TypeRef(
                        name=type_name,
                        kind="return",
                        line=return_type.start_point[0] + 1,
                    )
                )

        # Walk the function body for nested definitions and variable annotations.
        # Call extraction is handled once at the module level by ``parse()``.
        body = node.child_by_field_name("body")
        if body is not None:
            # Nested functions/classes inside a function are not methods,
            # so we pass class_name="" to keep them as standalone symbols.
            self._walk(body, content, result, class_name="")

    def _build_signature(self, func_node: Node, content: str) -> str:
        """Build a human-readable signature string for a function."""
        name_node = func_node.child_by_field_name("name")
        params_node = func_node.child_by_field_name("parameters")
        return_type = func_node.child_by_field_name("return_type")

        if name_node is None or params_node is None:
            return ""

        name = name_node.text.decode("utf8")
        params = params_node.text.decode("utf8")
        sig = f"def {name}{params}"

        if return_type is not None:
            sig += f" -> {return_type.text.decode('utf8')}"

        return sig

    def _extract_param_types(self, func_node: Node, result: ParseResult) -> None:
        """Extract type annotations from function parameters."""
        params_node = func_node.child_by_field_name("parameters")
        if params_node is None:
            return

        for param in params_node.children:
            if param.type == "typed_parameter":
                self._extract_typed_param(param, result)
            elif param.type == "typed_default_parameter":
                self._extract_typed_param(param, result)

    def _extract_typed_param(self, param_node: Node, result: ParseResult) -> None:
        """Extract a single typed parameter's type reference."""
        # The parameter name is the first identifier child.
        param_name = ""
        for child in param_node.children:
            if child.type == "identifier":
                param_name = child.text.decode("utf8")
                break

        type_node = param_node.child_by_field_name("type")
        if type_node is None:
            return

        type_name = self._extract_type_name(type_node)
        if type_name and type_name not in _BUILTIN_TYPES:
            result.type_refs.append(
                TypeRef(
                    name=type_name,
                    kind="param",
                    line=type_node.start_point[0] + 1,
                    param_name=param_name,
                )
            )

    # ------------------------------------------------------------------
    # Classes
    # ------------------------------------------------------------------

    def _extract_class(
        self,
        node: Node,
        content: str,
        result: ParseResult,
    ) -> None:
        """Extract a class definition and its contents."""
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return

        class_name = name_node.text.decode("utf8")
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        node_content = content[node.start_byte : node.end_byte]

        result.symbols.append(
            SymbolInfo(
                name=class_name,
                kind="class",
                start_line=start_line,
                end_line=end_line,
                content=node_content,
            )
        )

        # Extract inheritance (superclasses).
        superclasses = node.child_by_field_name("superclasses")
        if superclasses is not None:
            for child in superclasses.children:
                if child.is_named and child.type == "identifier":
                    parent_name = child.text.decode("utf8")
                    result.heritage.append((class_name, "extends", parent_name))

        # Walk into the class body to find methods and other definitions.
        body = node.child_by_field_name("body")
        if body is not None:
            self._walk(body, content, result, class_name=class_name)

    # ------------------------------------------------------------------
    # Imports
    # ------------------------------------------------------------------

    def _extract_import(self, node: Node, result: ParseResult) -> None:
        """Extract a plain ``import X`` statement."""
        # ``import_statement`` children: "import", dotted_name [, ",", dotted_name ...]
        for child in node.children:
            if child.type == "dotted_name":
                module = child.text.decode("utf8")
                # For ``import os.path`` the imported name available locally is "path"
                # (the last segment), but the module is the full dotted path.
                parts = module.split(".")
                result.imports.append(
                    ImportInfo(
                        module=module,
                        names=[parts[-1]],
                    )
                )
            elif child.type == "aliased_import":
                name_node = child.child_by_field_name("name")
                alias_node = child.child_by_field_name("alias")
                if name_node is not None:
                    module = name_node.text.decode("utf8")
                    parts = module.split(".")
                    alias = alias_node.text.decode("utf8") if alias_node else ""
                    result.imports.append(
                        ImportInfo(
                            module=module,
                            names=[parts[-1]],
                            alias=alias,
                        )
                    )

    def _extract_import_from(self, node: Node, result: ParseResult) -> None:
        """Extract a ``from X import Y`` statement."""
        module_name_node = node.child_by_field_name("module_name")
        if module_name_node is None:
            return

        is_relative = module_name_node.type == "relative_import"
        module = module_name_node.text.decode("utf8")

        # Collect all imported names (everything after ``import``).
        names: list[str] = []
        past_import = False
        for child in node.children:
            if child.type == "import":
                past_import = True
                continue
            if past_import and child.type == "dotted_name":
                names.append(child.text.decode("utf8"))

        result.imports.append(
            ImportInfo(
                module=module,
                names=names,
                is_relative=is_relative,
            )
        )

    # ------------------------------------------------------------------
    # Calls and variable annotations (extracted from bodies)
    # ------------------------------------------------------------------

    def _extract_annotations_from_expression(
        self,
        node: Node,
        result: ParseResult,
    ) -> None:
        """Extract variable annotations from an expression_statement."""
        for child in node.children:
            if child.type == "assignment":
                self._try_extract_variable_annotation(child, result)

    def _try_extract_variable_annotation(self, assignment_node: Node, result: ParseResult) -> None:
        """Extract a type reference from a variable annotation if present."""
        type_node = assignment_node.child_by_field_name("type")
        if type_node is None:
            return

        type_name = self._extract_type_name(type_node)
        if type_name and type_name not in _BUILTIN_TYPES:
            result.type_refs.append(
                TypeRef(
                    name=type_name,
                    kind="variable",
                    line=type_node.start_point[0] + 1,
                )
            )

    def _extract_calls_recursive(self, node: Node, result: ParseResult) -> None:
        """Recursively find and extract all call nodes."""
        if node.type == "call":
            self._extract_call(node, result)
            # Recurse into ALL children so we catch:
            # - nested calls in arguments: foo(bar())
            # - chained calls: obj.method1().method2() — the function part
            #   of the outer call contains the inner call node.
            for child in node.children:
                self._extract_calls_recursive(child, result)
            return

        for child in node.children:
            self._extract_calls_recursive(child, result)

    def _extract_call(self, call_node: Node, result: ParseResult) -> None:
        """Extract a single call node into a CallInfo."""
        func_node = call_node.child_by_field_name("function")
        if func_node is None:
            # Fallback: first named child is the function.
            for child in call_node.children:
                if child.is_named:
                    func_node = child
                    break
        if func_node is None:
            return

        line = call_node.start_point[0] + 1

        if func_node.type == "identifier":
            # Simple call: foo()
            result.calls.append(
                CallInfo(
                    name=func_node.text.decode("utf8"),
                    line=line,
                )
            )
        elif func_node.type == "attribute":
            # Method/attribute call: obj.method()
            name, receiver = self._extract_attribute_call(func_node)
            result.calls.append(
                CallInfo(
                    name=name,
                    line=line,
                    receiver=receiver,
                )
            )

    def _extract_attribute_call(self, attr_node: Node) -> tuple[str, str]:
        """Extract (method_name, receiver) from an attribute node.

        For chained calls like ``obj.method1().method2()``, the outer call's
        function is ``attribute(call(...), "method2")``.  We extract
        ``method2`` as the name and the first identifier in the chain as
        the receiver.
        """
        # The last identifier child of the attribute node is the method name.
        method_name = ""
        for child in reversed(attr_node.children):
            if child.type == "identifier":
                method_name = child.text.decode("utf8")
                break

        # The receiver is the object part (first child of attribute).
        receiver = ""
        obj_node = attr_node.children[0] if attr_node.children else None
        if obj_node is not None:
            if obj_node.type == "identifier":
                receiver = obj_node.text.decode("utf8")
            elif obj_node.type == "attribute":
                # Nested attribute access like ``self.logger.info()`` — use the root.
                receiver = self._root_identifier(obj_node)
            elif obj_node.type == "call":
                # Chained call like ``get_user().save()`` — try the innermost identifier.
                receiver = self._root_identifier(obj_node)

        return method_name, receiver

    def _root_identifier(self, node: Node) -> str:
        """Walk down into the leftmost identifier of an expression."""
        current = node
        while current is not None:
            if current.type == "identifier":
                return current.text.decode("utf8")
            if current.children:
                current = current.children[0]
            else:
                break
        return ""

    # ------------------------------------------------------------------
    # Type name helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_type_name(type_node: Node) -> str:
        """Extract the primary type name from a type annotation node.

        For simple types like ``User``, returns ``"User"``.
        For generic types like ``list[User]``, returns ``"list"``.
        For complex types, returns the text of the first identifier found.
        """
        # The ``type`` wrapper node contains the actual type child.
        if type_node.type == "type" and type_node.children:
            inner = type_node.children[0]
            if inner.type == "identifier":
                return inner.text.decode("utf8")
            if inner.type == "generic_type":
                # e.g., ``Optional[User]`` — return "Optional"
                for child in inner.children:
                    if child.type == "identifier":
                        return child.text.decode("utf8")
            # Fallback: return text of first identifier found anywhere.
            return PythonParser._find_first_identifier(inner)
        if type_node.type == "identifier":
            return type_node.text.decode("utf8")
        return PythonParser._find_first_identifier(type_node)

    @staticmethod
    def _find_first_identifier(node: Node) -> str:
        """DFS for the first identifier node."""
        if node.type == "identifier":
            return node.text.decode("utf8")
        for child in node.children:
            found = PythonParser._find_first_identifier(child)
            if found:
                return found
        return ""
